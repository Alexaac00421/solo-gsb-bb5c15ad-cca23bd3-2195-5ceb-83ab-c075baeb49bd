"""FastAPI 入口：HTTP API + 操作页面静态托管。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    BlockingConstraint,
    ScheduleDiffRequest,
    ScheduleDiffResponse,
    ScheduleItem,
    ScheduleRequest,
    ScheduleResponse,
)
from .diff import build_diff, run_request
from .times import to_local_display

app = FastAPI(
    title="跨时区发布窗口编排",
    description="在时区/DST、依赖先后、禁发、冻结与资源约束下求可执行发布排程",
    version="1.0.0",
)

WEB_DIR = Path(__file__).parent / "web"


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/schedule", response_model=ScheduleResponse)
def schedule(req: ScheduleRequest) -> ScheduleResponse:
    run = run_request(req)

    if run.empty:
        return ScheduleResponse(
            feasible=False, status="infeasible",
            blockers=[BlockingConstraint(code="empty_services",
                                        message="至少需要一个服务")],
        )

    if run.validation_errors is not None:
        # 422：输入本身有问题（DST 重复/不存在、坏时区、依赖环、区间非法等）。
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=422,
                            content={"errors": run.validation_errors})

    assert run.norm is not None and run.result is not None
    norm, result = run.norm, run.result

    if not result.feasible:
        return ScheduleResponse(
            feasible=False,
            status=result.status,
            blockers=[BlockingConstraint(**b) for b in (result.blockers or [])],
        )

    return ScheduleResponse(
        feasible=True, status=result.status,
        schedule=[_build_item(norm, sid, st)
                  for sid, st in sorted(result.starts.items(), key=lambda kv: kv[1])],
    )


def _build_item(norm, sid: str, st: int) -> ScheduleItem:
    s = norm.services[sid]
    end = st + s.duration
    occ = st + s.duration + s.rollback
    used = next(
        (w for w in s.windows if w.start <= st and occ <= w.end),
        None,
    )
    window_text = (
        f"{_iso(used.start)} ~ {_iso(used.end)} UTC（{used.label}）"
        if used else "（窗口信息缺失）"
    )
    return ScheduleItem(
        service_id=sid,
        name=s.name,
        deploy_start_utc=_iso(st),
        deploy_end_utc=_iso(end),
        occupied_until_utc=_iso(occ),
        local_time=to_local_display(datetime.fromtimestamp(st, tz=timezone.utc), s.timezone),
        timezone=s.timezone,
        window_used=window_text,
        resources=s.resources,
    )


@app.post("/api/schedule/diff", response_model=ScheduleDiffResponse)
def schedule_diff(body: ScheduleDiffRequest):
    """变更影响预演：分别求解 current / modified，按服务 ID 对齐后返回时刻差异。

    任一请求输入错误或无解时，只返回对应一侧的完整错误/阻塞约束组合，不输出差异；
    两份请求都无变化时返回空差异（changed/added/removed 均为空列表）。
    """
    current = run_request(body.current)
    modified = run_request(body.modified)

    resp = build_diff(current, modified)
    if not resp.feasible and resp.failures:
        # 只要存在输入错误（422 类）即以 422 返回；纯无解/超时仍为 200。
        if any(f.kind == "validation" for f in resp.failures.values()):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=422,
                content=resp.model_dump(mode="json"),
            )
    return resp


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
