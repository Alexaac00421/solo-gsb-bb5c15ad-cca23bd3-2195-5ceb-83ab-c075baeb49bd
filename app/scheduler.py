"""CP-SAT 调度引擎。

建模（全部时间为 epoch 秒）：
- 每个服务一个部署起点 start；部署区间 [start, start+deploy)，
  占用区间 [start, start+deploy+rollback)。
- 起点必须落在候选时间窗的并集内：对并集在总时间轴上的每个“补集缺口”建立
  NoOverlap（占用区间 vs 缺口）。
- 维护冻结：占用区间不得与任何冻结区间相交（部署与回滚窗口都要避开）。
- 团队禁发：仅部署区间不得与所属团队禁发时段相交（回滚允许进入禁发期）。
- 共享资源：同一资源任意时刻的占用数不得超过其容量（默认 1，即两两不重叠；
  resource_capacities 可声明更大的正整数容量，允许至多容量个服务并行；
  resource_capacity_periods 可声明临时容量时段（左闭右开），时段内按当时容量、
  时段外按基准容量，容量 0 表示该时段不可占用）。
- 依赖：downstream.start >= upstream.start + deploy_up + gap。

CP-SAT 的 NoOverlap 不接受约束级 enforcement literal，因此采用“可选镜像区间”：
每个受假设控制的约束都有一个布尔字面量，把参与区间建成以该字面量为 presence 的
可选区间（固定时段则建常量轴上的可选固定区间）。字面量为假时区间缺席，
NoOverlap 自动忽略；求解时把所有字面量加入假设（assumptions）。
不可行时 CP-SAT 返回足以导致矛盾的字面量子集（sufficient assumption core），
即“阻塞它的约束组合”，而不会给出任何部分排程。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from ortools.sat.python import cp_model

from .validator import Normalized


@dataclass
class SolveResult:
    feasible: bool
    status: str  # optimal | feasible | infeasible | unknown
    starts: dict[str, int] | None
    blockers: list[dict] | None


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    out = [list(ordered[0])]
    for a, b in ordered[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _complement(union: list[tuple[int, int]], h0: int, h1: int) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    cur = h0
    for a, b in union:
        if a > cur:
            gaps.append((cur, a))
        cur = max(cur, b)
    if cur < h1:
        gaps.append((cur, h1))
    return gaps


def _capacity_message(resource: str, cap: int, users: list[str], services) -> str:
    """基准容量约束的阻塞说明（与无临时容量时段时的措辞一致）。"""
    if cap == 1:
        return (
            f"共享资源 {resource} 同时只能被一个服务占用（部署+回滚窗口）；"
            f"涉及: {', '.join(services[u].name for u in users)}"
        )
    return (
        f"共享资源 {resource} 的并行容量为 {cap}（部署+回滚窗口内任意时刻至多 {cap} 个服务占用），"
        f"但共有 {len(users)} 个服务需要占用: {', '.join(services[u].name for u in users)}；"
        f"超出容量的服务必须错峰"
    )


def _period_message(resource: str, period, users: list[str], services) -> str:
    """临时容量时段约束的阻塞说明。"""
    span = f"{_iso(period.start)} ~ {_iso(period.end)} UTC"
    names = ", ".join(services[u].name for u in users)
    if period.capacity == 0:
        return (
            f"共享资源 {resource} 在 {span} 期间临时限流为 0（不可占用）；"
            f"涉及服务的部署+回滚窗口不得与该时段重叠: {names}"
        )
    return (
        f"共享资源 {resource} 在 {span} 期间容量临时为 {period.capacity}"
        f"（部署+回滚窗口内任意时刻至多 {period.capacity} 个服务占用）；涉及: {names}"
    )


def _add_segment_capacity(m: cp_model.CpModel, starts, services, users: list[str],
                          a: int, b: int, cap: int, lit, h0: int, h1: int,
                          prefix: str) -> None:
    """约束：分段 [a, b) 内任意时刻的占用并发数 ≤ cap（左闭右开）。

    对每个占用区间 [st, st+dur) 建“钳制到分段”的可选区间：仅当占用与分段重叠时
    出席（ov），其区间为 [max(st,a), min(st+dur,b))。对该分段做 Cumulative(cap)
    即等价于分段内任意时刻并发 ≤ cap；降容时刻落在既有占用期间时，该占用在分段内
    仍有非空钳制区间，计入当时容量。lit 为假时全部钳制区间缺席，约束整体放松。
    """
    clamped = []
    for u in users:
        dur = services[u].duration + services[u].rollback
        start = starts[u]
        # ov ⟺ 占用与分段重叠 ⟺ st ≤ b-1 且 st ≥ a-dur+1（整数秒）
        ge = m.new_bool_var(f"{prefix}_{u}_ge")
        le = m.new_bool_var(f"{prefix}_{u}_le")
        m.add(start >= a - dur + 1).only_enforce_if(ge)
        m.add(start <= a - dur).only_enforce_if(ge.Not())
        m.add(start <= b - 1).only_enforce_if(le)
        m.add(start >= b).only_enforce_if(le.Not())
        ov = m.new_bool_var(f"{prefix}_{u}_ov")
        m.add_bool_and([ge, le]).only_enforce_if(ov)
        m.add_bool_or([ge.Not(), le.Not()]).only_enforce_if(ov.Not())
        # 钳制区间 [s2, s2+sz)：s2 = max(st, a)，e2 = min(st+dur, b)，sz = max(0, e2-s2)
        s2 = m.new_int_var(h0, h1, f"{prefix}_{u}_s")
        m.add_max_equality(s2, [start, a])
        e2 = m.new_int_var(h0, h1, f"{prefix}_{u}_e")
        m.add_min_equality(e2, [start + dur, b])
        sz = m.new_int_var(0, h1 - h0, f"{prefix}_{u}_sz")
        m.add_max_equality(sz, [e2 - s2, 0])
        end2 = m.new_int_var(h0, h1, f"{prefix}_{u}_end")
        m.add(end2 == s2 + sz)
        pres = m.new_bool_var(f"{prefix}_{u}_pres")
        m.add_multiplication_equality(pres, [ov, lit])
        clamped.append(
            m.new_optional_interval_var(s2, sz, end2, pres, f"{prefix}_{u}_iv"))
    m.add_cumulative(clamped, [1] * len(clamped), cap)


def solve(norm: Normalized) -> SolveResult:
    m = cp_model.CpModel()
    services = norm.services

    h0 = min(w.start for s in services.values() for w in s.windows)
    h1 = max(w.end for s in services.values() for w in s.windows)

    # 已确认（锁定）时刻可能落在任何候选窗之外——这本身就是需要阻塞核解释的矛盾。
    # 因此把建模时间轴延伸到所有锁定时刻（及其占用结束），锁定等式仍作为带假设字面量的
    # 普通约束参与求解；锁点与窗口/冻结/禁发/资源/依赖冲突时，lock 字面量会与相关
    # 约束一起进入 sufficient-assumption 阻塞核。
    H0 = min([h0, *norm.pinned_starts.values()] if norm.pinned_starts else [h0])
    H1 = h1
    for sid, pst in norm.pinned_starts.items():
        H1 = max(H1, pst + services[sid].duration + services[sid].rollback)

    # 假设字面量 index -> (code, message)，用于不可行核解释。
    assumption_map: dict[int, tuple[str, str]] = {}

    starts: dict[str, cp_model.IntVar] = {}
    deploy_iv: dict[str, cp_model.IntervalVar] = {}
    occupy_iv: dict[str, cp_model.IntervalVar] = {}

    for sid, s in services.items():
        start = m.new_int_var(H0, H1, f"start_{sid}")
        starts[sid] = start
        deploy_iv[sid] = m.new_fixed_size_interval_var(start, s.duration, f"deploy_{sid}")
        occupy_iv[sid] = m.new_fixed_size_interval_var(
            start, s.duration + s.rollback, f"occupy_{sid}")
        # 窗口之外的缺口仅在 H0~H1 内部建模，而最后的窗口终点即 h1；因此还须显式保证
        # 占用区间（含回滚）不越过 H1，否则起点取在 H1 时整段占用会漏到所有窗口之外。
        m.add(start + s.duration + s.rollback <= H1)

    def new_assumption(code: str, message: str) -> cp_model.IntVar:
        b = m.new_bool_var(f"assm_{code.replace(':', '_')}")
        m.add_assumptions([b])
        assumption_map[b.index] = (code, message)
        return b

    def opt_mirror(base_iv: cp_model.IntervalVar, lit, name: str) -> cp_model.IntervalVar:
        """以同一 start/size/end + presence=lit 建立可选镜像区间。

        注意 CP-SAT 的参数顺序是 (start, size, end, is_present, name)。
        """
        return m.new_optional_interval_var(
            base_iv.start_expr(), base_iv.size_expr(), base_iv.end_expr(), lit, name)

    def opt_constant(a: int, size: int, lit, name: str) -> cp_model.IntervalVar:
        """常量轴 [a, a+size) 上的可选固定区间。"""
        return m.new_optional_fixed_size_interval_var(m.new_constant(a), size, lit, name)

    # ---- 已确认服务：必须在锁定时刻准时部署 ----
    # 锁定是带假设字面量的硬等式：锁点与任何其他约束冲突时，lock 字面量会与相关
    # 约束字面量一起出现在 sufficient-assumption 阻塞核中；锁定本身不豁免任何约束。
    for sid, pst in norm.pinned_starts.items():
        lit = new_assumption(
            f"lock:{sid}",
            f"服务 {services[sid].name}（{sid}）已确认必须在 {_iso(pst)} UTC 准时部署"
            "（锁定开始时刻不可移动；锁定不豁免候选窗、依赖、团队禁发、维护冻结与资源容量约束）",
        )
        m.add(starts[sid] == pst).only_enforce_if(lit)

    # ---- 候选时间窗：占用区间必须整体落在窗口并集内 ----
    for sid, s in services.items():
        union = _merge_intervals([(w.start, w.end) for w in s.windows])
        gaps = _complement(union, H0, H1)
        for g_start, g_end in gaps:
            lit = new_assumption(
                f"window:{sid}:{g_start}",
                f"服务 {s.name}（{sid}）的部署与回滚必须落在其候选时间窗内；"
                f"窗口外时段 {_iso(g_start)} ~ {_iso(g_end)} UTC 不可发布",
            )
            m.add_no_overlap([
                opt_mirror(occupy_iv[sid], lit, f"win_occ_{sid}_{g_start}"),
                opt_constant(g_start, g_end - g_start, lit, f"win_gap_{sid}_{g_start}"),
            ])

    # ---- 全局维护冻结：占用区间（含回滚窗口）避让 ----
    for f in norm.freezes:
        for sid, s in services.items():
            lit = new_assumption(
                f"freeze:{f.label}:{sid}",
                f"{f.label}（{_iso(f.start)} ~ {_iso(f.end)} UTC）期间"
                f"服务 {s.name}（{sid}）不得部署或处于回滚窗口内",
            )
            m.add_no_overlap([
                opt_mirror(occupy_iv[sid], lit, f"fz_occ_{sid}_{f.start}"),
                opt_constant(f.start, f.end - f.start, lit, f"fz_f_{sid}_{f.start}"),
            ])

    # ---- 团队禁发：仅部署区间避让（回滚允许进入） ----
    for sid, teams in norm.team_members.items():
        for team in teams:
            for p in norm.blackouts.get(team, []):
                lit = new_assumption(
                    f"blackout:{team}:{sid}:{p.start}",
                    f"团队 {team} 禁发（{_iso(p.start)} ~ {_iso(p.end)} UTC）期间"
                    f"服务 {services[sid].name}（{sid}）不得开始部署",
                )
                m.add_no_overlap([
                    opt_mirror(deploy_iv[sid], lit, f"bo_dep_{sid}_{p.start}"),
                    opt_constant(p.start, p.end - p.start, lit, f"bo_p_{sid}_{p.start}"),
                ])

    # ---- 共享资源占用：同资源任意时刻的并行占用数不得超过当时容量 ----
    # 基准容量来自 resource_capacities（未声明为 1）；resource_capacity_periods 声明的
    # 临时容量时段（左闭右开）在时段内覆盖基准容量，时段外仍按基准容量。
    resource_users: dict[str, list[str]] = {}
    for sid, s in services.items():
        for r in s.resources:
            resource_users.setdefault(r, []).append(sid)
    for resource, users in resource_users.items():
        cap = norm.resource_capacities.get(resource, 1)
        periods = norm.capacity_periods.get(resource, [])
        if not periods:
            # 无临时容量时段：恒定容量，沿用 NoOverlap（容量 1）/ Cumulative（容量 ≥ 2）。
            if len(users) <= cap:
                # 用户数本就不超过容量：任何时刻都不可能超载，无需建立约束。
                continue
            lit = new_assumption(f"resource:{resource}",
                                 _capacity_message(resource, cap, users, services))
            mirrors = [opt_mirror(occupy_iv[u], lit, f"res_{resource}_{u}") for u in users]
            if cap == 1:
                m.add_no_overlap(mirrors)
            else:
                # 每个占用对资源的需求为 1；可选区间在 lit 为假时缺席、需求自动归零，
                # 因此不可行核可以像 NoOverlap 一样直接定位到该资源假设。
                m.add_cumulative(mirrors, [1] * len(mirrors), cap)
            continue

        # 有临时容量时段：把总时间轴按时段边界切分为若干容量恒定的分段，对每个
        # “当时容量 < 占用服务数”的分段约束其内部任意时刻的占用并发数不超过当时容量。
        clipped = sorted(
            ((max(p.start, H0), min(p.end, H1), p) for p in periods
             if max(p.start, H0) < min(p.end, H1)),
            key=lambda t: (t[0], t[1]),
        )
        bounds = sorted({H0, H1} | {a for a, _, _ in clipped} | {b for _, b, _ in clipped})
        base_lit: cp_model.IntVar | None = None
        period_lits: dict[int, cp_model.IntVar] = {}
        for a, b in zip(bounds, bounds[1:]):
            covering = next((p for ca, cb, p in clipped if ca <= a and b <= cb), None)
            seg_cap = covering.capacity if covering is not None else cap
            if seg_cap >= len(users):
                # 该分段内容量本就够用：任何时刻都不可能超载，无需建立约束。
                continue
            if covering is None:
                if base_lit is None:
                    base_lit = new_assumption(
                        f"resource:{resource}",
                        _capacity_message(resource, cap, users, services))
                lit = base_lit
            else:
                if covering.start not in period_lits:
                    period_lits[covering.start] = new_assumption(
                        f"resource:{resource}:{covering.start}",
                        _period_message(resource, covering, users, services))
                lit = period_lits[covering.start]
            _add_segment_capacity(m, starts, services, users, a, b, seg_cap, lit,
                                  H0, H1, f"res_{resource}_{a}")

    # ---- 依赖先后 + 最小间隔（线性不等式支持直接挂 literal） ----
    for d in norm.dependencies:
        up, dn = services[d.upstream], services[d.downstream]
        lit = new_assumption(
            f"dependency:{d.upstream}->{d.downstream}",
            f"依赖顺序：{dn.name}（{d.downstream}）必须在 {up.name}（{d.upstream}）"
            f"部署完成" + (f"并再等待 {d.gap // 60} 分钟" if d.gap else "") + "之后才能开始",
        )
        m.add(
            starts[d.downstream] >= starts[d.upstream] + up.duration + d.gap
        ).only_enforce_if(lit)

    # ---- 目标 ----
    # 无参考时刻时沿用原目标：所有服务部署开始时刻之和最小（尽量早、可复现）。
    # 给出参考时刻时采用三层词典序目标，逐层分别求解且每层都必须证明最优：
    #   1) 最小化偏离参考时刻的服务数（未给参考时刻的服务不参与本层）；
    #   2) 在 1) 最优的前提下最小化偏移总时长（秒，早于/晚于均计为绝对偏差）；
    #   3) 在前两层最优的前提下最小化所有服务部署开始时刻之和（原“尽早发布”目标，
    #      未给参考时刻的服务只在本层参与）。
    # 参考时刻仅是优化目标，不构成任何约束：参考点落在窗外/冻结期/被资源占满时
    # 服务照样排不进该时刻（偏离指标取 1），约束永远不会被豁免。
    early_expr = sum(starts.values())

    ref_exprs: dict[str, tuple[cp_model.IntVar, cp_model.IntVar]] = {}
    if norm.reference_starts:
        for sid, ref in norm.reference_starts.items():
            # 参考点本身不受约束，可能远离建模轴（如落在全部候选窗之外）；
            # 偏差变量上界须覆盖起点域 [H0, H1] 到参考点的最大距离，否则会伪不可行。
            dev_bound = max(abs(H1 - ref), abs(H0 - ref))
            dev = m.new_int_var(0, dev_bound, f"refdev_{sid}")
            m.add_abs_equality(dev, starts[sid] - ref)
            off = m.new_bool_var(f"refoff_{sid}")
            m.add(dev >= 1).only_enforce_if(off)
            m.add(dev == 0).only_enforce_if(off.Not())
            ref_exprs[sid] = (off, dev)
        count_expr = sum(off for off, _ in ref_exprs.values())
        total_dev_expr = sum(dev for _, dev in ref_exprs.values())

    def run_solve(deadline: float):
        # 共享总时限：每层最多用到 deadline 为止；剩余极少时给 10ms 下限，避免 0 时长。
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(0.01, deadline - time.monotonic())
        solver.parameters.num_search_workers = 8
        status = solver.solve(m)
        return status, solver

    def infeasible_result(solver) -> SolveResult:
        core_literals = solver.sufficient_assumptions_for_infeasibility()
        blockers = []
        if core_literals:
            for lit in core_literals:
                code, message = assumption_map[lit]
                blockers.append({"code": code, "message": message})
        else:
            blockers = [{"code": code, "message": msg}
                        for code, msg in assumption_map.values()]
        return SolveResult(False, "infeasible", None, blockers)

    def unknown_result(layered: bool = False) -> SolveResult:
        if layered:
            message = (
                "求解器在时限内未能证明分层目标（偏离参考时刻的服务数 → "
                "偏移总时长 → 尽早发布）的最优性，按约定不输出排程；"
                "可增大 max_solve_seconds 后重试"
            )
        else:
            message = (
                "求解器在时限内未能判定可行性（未证明最优），"
                "可增大 max_solve_seconds 后重试"
            )
        return SolveResult(
            False, "unknown", None,
            [{"code": "solver_unknown", "message": message}],
        )

    deadline = time.monotonic() + norm.max_solve_seconds

    if not norm.reference_starts:
        # 原行为：单层“尽早发布”；FEASIBLE（未证明最优）也照常返回。
        m.minimize(early_expr)
        status, solver = run_solve(deadline)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return SolveResult(
                feasible=True,
                status="optimal" if status == cp_model.OPTIMAL else "feasible",
                starts={sid: solver.value(v) for sid, v in starts.items()},
                blockers=None,
            )
        if status == cp_model.INFEASIBLE:
            return infeasible_result(solver)
        return unknown_result()

    # 分层优化：每一层都必须在共享时限内证明 OPTIMAL，否则 unknown（不输出排程）。
    # 第 1 层：偏离参考时刻的服务数最少。
    m.minimize(count_expr)
    status, solver = run_solve(deadline)
    if status == cp_model.INFEASIBLE:
        return infeasible_result(solver)
    if status != cp_model.OPTIMAL:
        return unknown_result(layered=True)
    best_count = int(round(solver.objective_value))
    m.add(count_expr <= best_count)

    # 第 2 层：偏移总时长最短。
    m.minimize(total_dev_expr)
    status, solver = run_solve(deadline)
    if status == cp_model.INFEASIBLE:
        return infeasible_result(solver)
    if status != cp_model.OPTIMAL:
        return unknown_result(layered=True)
    best_dev = int(round(solver.objective_value))
    m.add(total_dev_expr <= best_dev)

    # 第 3 层：尽早发布（所有服务开始时刻之和最小）。
    m.minimize(early_expr)
    status, solver = run_solve(deadline)
    if status == cp_model.INFEASIBLE:
        return infeasible_result(solver)
    if status != cp_model.OPTIMAL:
        return unknown_result(layered=True)

    return SolveResult(
        feasible=True,
        status="optimal",
        starts={sid: solver.value(v) for sid, v in starts.items()},
        blockers=None,
    )
