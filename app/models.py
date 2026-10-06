"""API 输入/输出数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class WindowIn(BaseModel):
    start: str = Field(..., description="ISO 8601 时间；不带偏移量时按时区字段解析")
    end: str
    timezone: str = Field("UTC", description="IANA 时区，如 Asia/Shanghai")


class ServiceIn(BaseModel):
    id: str
    name: str = ""
    timezone: str = Field("UTC", description="排程结果的本地展示时区")
    duration_minutes: int = Field(..., gt=0, description="部署占用时长")
    rollback_minutes: int = Field(0, ge=0, description="部署后必须可回滚的资源占用时长")
    resources: list[str] = Field(default_factory=list, description="占用的共享资源名")
    windows: list[WindowIn] = Field(..., min_length=1, description="候选发布时间窗")


class DependencyIn(BaseModel):
    upstream: str
    downstream: str
    gap_minutes: int = Field(0, ge=0, description="上游部署结束到下游部署开始的最小间隔")


class BlackoutIn(BaseModel):
    team: str = Field(..., description="团队名")
    start: str
    end: str
    timezone: str = "UTC"


class FreezeIn(BaseModel):
    label: str = Field(..., description="冻结名称，如“双十一封网”")
    start: str
    end: str
    timezone: str = "UTC"


class ServiceTeamIn(BaseModel):
    service: str
    team: str


class CapacityPeriodIn(BaseModel):
    resource: str = Field(..., description="资源名")
    start: str = Field(..., description="ISO 8601 时间；不带偏移量时按时区字段解析")
    end: str
    timezone: str = Field("UTC", description="IANA 时区，如 Asia/Shanghai")
    capacity: int = Field(..., description="该时段内的并行容量（非负整数，0 表示该时段不可占用）")


class PinnedStartIn(BaseModel):
    service: str = Field(..., description="已确认（锁定）服务的 ID；必须在指定时刻准时部署")
    start: str = Field(..., description="已确认的部署开始时刻（ISO 8601；不带偏移量时按时区字段解析，"
                                        "沿用现有 DST 重复/不存在校验）")
    timezone: str = Field("UTC", description="start 为不带偏移量的墙上时间时使用的 IANA 时区")


class ReferenceStartIn(BaseModel):
    service: str = Field(..., description="给出参考开始时刻的服务 ID（未确认服务）")
    start: str = Field(..., description="参考部署开始时刻（仅用于分层优化目标，不豁免候选窗、依赖、"
                                        "禁发、冻结、资源容量等任何约束）")
    timezone: str = Field("UTC", description="start 为不带偏移量的墙上时间时使用的 IANA 时区")


class ScheduleRequest(BaseModel):
    services: list[ServiceIn]
    dependencies: list[DependencyIn] = Field(default_factory=list)
    service_teams: list[ServiceTeamIn] = Field(default_factory=list)
    team_blackouts: list[BlackoutIn] = Field(default_factory=list)
    freezes: list[FreezeIn] = Field(default_factory=list)
    resource_capacities: dict[str, int] = Field(
        default_factory=dict,
        description="共享资源并行容量；未声明的资源容量为 1。同一时刻占用数不得超过容量",
    )
    resource_capacity_periods: list[CapacityPeriodIn] = Field(
        default_factory=list,
        description="共享资源的临时容量时段（左闭右开）；时段内在任意时刻占用数不得超过该时段容量"
        "（0 表示不可占用），时段外沿用 resource_capacities，未声明基准容量为 1",
    )
    pinned_starts: list[PinnedStartIn] = Field(
        default_factory=list,
        description="已确认服务的锁定部署开始时刻：这些服务必须在指定时刻准时部署；"
        "锁定不豁免候选窗、依赖、团队禁发、维护冻结与资源分时容量等任何约束，"
        "冲突时 blockers 会返回包含 lock 条件的阻塞组合，不返回部分排程",
    )
    reference_starts: list[ReferenceStartIn] = Field(
        default_factory=list,
        description="未确认服务的参考部署开始时刻：仅参与分层目标（先最小化偏离参考时刻的服务数，"
        "再最小化偏移总秒数，最后最小化部署开始时刻之和），不豁免任何约束；"
        "未给出参考时刻的服务只参与最后一层（尽早发布）目标",
    )
    max_solve_seconds: float = Field(10.0, gt=0, le=60)


class ScheduleItem(BaseModel):
    service_id: str
    name: str
    deploy_start_utc: str
    deploy_end_utc: str
    occupied_until_utc: str
    local_time: str
    timezone: str
    window_used: str
    resources: list[str]


class BlockingConstraint(BaseModel):
    code: str
    message: str


class ScheduleResponse(BaseModel):
    feasible: bool
    schedule: list[ScheduleItem] | None = None
    blockers: list[BlockingConstraint] | None = None
    status: str  # "optimal" | "feasible" | "infeasible" | "unknown"


# ---- 变更影响预演（/api/schedule/diff） ----


class ScheduleDiffRequest(BaseModel):
    current: ScheduleRequest = Field(..., description="当前（变更前）排程请求")
    modified: ScheduleRequest = Field(..., description="修改后（变更后）的排程请求")


class TimeChange(BaseModel):
    before_utc: str
    after_utc: str
    delta_minutes: float = Field(..., description="after - before，正数表示推迟，负数表示提前")


class ServiceChange(BaseModel):
    service_id: str
    name: str
    deploy_start: TimeChange | None = Field(None, description="部署开始时刻的变化；未变化为 null")
    occupied_until: TimeChange | None = Field(
        None, description="占用结束时刻（部署结束 + 回滚窗口）的变化；未变化为 null")


class ServiceRef(BaseModel):
    service_id: str
    name: str


class ScheduleDiff(BaseModel):
    changed: list[ServiceChange] = Field(..., description="两侧都存在、开始或占用结束时刻发生变化的服务")
    added: list[ServiceRef] = Field(..., description="仅在 modified 中出现的服务")
    removed: list[ServiceRef] = Field(..., description="仅在 current 中出现的服务")


class ValidationError(BaseModel):
    code: str
    field: str | None = None
    message: str
    suggestions: list[str] | None = None


class RequestFailure(BaseModel):
    kind: str  # "validation" | "infeasible" | "unknown"
    status: str | None = None  # 求解器状态（infeasible/unknown）
    errors: list[ValidationError] | None = None  # kind=validation 时的完整输入错误
    blockers: list[BlockingConstraint] | None = None  # kind=infeasible/unknown 时的阻塞约束组合


class ScheduleDiffResponse(BaseModel):
    feasible: bool
    diff: ScheduleDiff | None = None
    failures: dict[str, RequestFailure] | None = Field(
        None, description="任一请求输入错误或无解时，按 current/modified 给出对应错误，不输出差异")
