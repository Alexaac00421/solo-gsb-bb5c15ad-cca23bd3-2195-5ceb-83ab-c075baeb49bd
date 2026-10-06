"""变更影响预演：分别求解两份排程请求，按服务 ID 对齐结果求差异。

规则：
- 任一请求输入错误（校验失败）或无解（infeasible/unknown）时，只返回对应一侧的
  完整错误或阻塞约束组合，不输出差异结果；
- 两份请求都可行时，按服务 ID 对齐两侧排程，列出：
  * changed：两侧都存在但部署开始或占用结束（部署 + 回滚）UTC 时刻发生变化的服务；
  * added：仅在新请求（modified）中出现的服务；
  * removed：仅在当前请求（current）中出现的服务；
- 两份请求都无变化时，changed/added/removed 均为空列表。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .models import (
    ScheduleDiff,
    ScheduleDiffResponse,
    ScheduleRequest,
    ServiceChange,
    ServiceRef,
    TimeChange,
)
from .scheduler import SolveResult, solve
from .validator import Normalized, validate_and_normalize


@dataclass
class RunOutcome:
    """单次排程请求的完整执行结果。

    四种互斥情形：
    - norm 非空且求解可行；
    - validation_errors 非空：输入错误（422）；
    - blockers 非空且 status 为 infeasible/unknown：无解/超时。
    """

    norm: Normalized | None = None
    result: SolveResult | None = None
    validation_errors: list[dict] | None = None
    empty: bool = False

    @property
    def feasible(self) -> bool:
        return not self.empty and not self.validation_errors and (
            self.result is not None and self.result.feasible)


def run_request(req: ScheduleRequest) -> RunOutcome:
    if not req.services:
        return RunOutcome(empty=True)

    norm, errors = validate_and_normalize(req)
    if errors:
        return RunOutcome(validation_errors=errors)

    return RunOutcome(norm=norm, result=solve(norm))


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _time_change(before: int, after: int) -> TimeChange:
    return TimeChange(
        before_utc=_iso(before),
        after_utc=_iso(after),
        delta_minutes=(after - before) / 60.0,
    )


def build_diff(current: RunOutcome, modified: RunOutcome) -> ScheduleDiffResponse:
    """两侧都可行时按服务 ID 对齐求差异，否则返回各侧失败信息。"""
    failures: dict[str, dict] = {}

    for label, run in (("current", current), ("modified", modified)):
        if run.empty:
            failures[label] = {
                "kind": "infeasible",
                "status": "infeasible",
                "blockers": [{"code": "empty_services",
                              "message": "至少需要一个服务"}],
            }
        elif run.validation_errors is not None:
            failures[label] = {
                "kind": "validation",
                "errors": run.validation_errors,
            }
        elif run.result is not None and not run.result.feasible:
            failures[label] = {
                "kind": "infeasible" if run.result.status == "infeasible" else "unknown",
                "status": run.result.status,
                "blockers": run.result.blockers or [],
            }

    if failures:
        # 任一请求失败：不输出差异，只给对应一侧的完整错误/阻塞组合。
        return ScheduleDiffResponse(feasible=False, failures=failures)

    assert current.norm and modified.norm and current.result and modified.result
    cur_svcs, new_svcs = current.norm.services, modified.norm.services
    cur_starts, new_starts = current.result.starts, modified.result.starts

    def occupy(norm: Normalized, sid: str, start: int) -> int:
        s = norm.services[sid]
        return start + s.duration + s.rollback

    cur_ids, new_ids = set(cur_starts), set(new_starts)

    changed: list[ServiceChange] = []
    for sid in sorted(cur_ids & new_ids):
        start_before, start_after = cur_starts[sid], new_starts[sid]
        occ_before = occupy(current.norm, sid, start_before)
        occ_after = occupy(modified.norm, sid, start_after)
        if start_before == start_after and occ_before == occ_after:
            continue
        changed.append(ServiceChange(
            service_id=sid,
            name=new_svcs[sid].name,
            deploy_start=(
                _time_change(start_before, start_after)
                if start_before != start_after else None),
            occupied_until=(
                _time_change(occ_before, occ_after)
                if occ_before != occ_after else None),
        ))

    added = [ServiceRef(service_id=sid, name=new_svcs[sid].name)
             for sid in sorted(new_ids - cur_ids)]
    removed = [ServiceRef(service_id=sid, name=cur_svcs[sid].name)
               for sid in sorted(cur_ids - new_ids)]

    return ScheduleDiffResponse(
        feasible=True,
        diff=ScheduleDiff(changed=changed, added=added, removed=removed),
    )
