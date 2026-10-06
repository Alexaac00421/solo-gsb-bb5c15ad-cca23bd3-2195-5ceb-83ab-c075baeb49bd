"""时区解析与夏令时（DST）本地时间分类。

规则：
- 带上 UTC 偏移量的时间串（如 2027-03-14T02:30:00-04:00 或 ...Z）视为绝对时刻，
  时区字段仅作展示用途，不参与 DST 判定。
- 不带偏移量的“墙上时间”必须配合 IANA 时区（如 Asia/Shanghai）解析：
  * 普通时间：fold=0 与 fold=1 得到同一个 UTC 时刻；
  * 重复时间（秋令时回拨，同一本地时间发生两次）：两种 fold 得到两个不同 UTC 时刻，
    且 fold=0 的偏移量更大；
  * 不存在时间（春令时前拨，该墙上时间被跳过）：两种 fold 不同，且 fold=0 偏移量更小。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class TimeParseError(ValueError):
    """时间串无法解析。"""


class UnknownTimezoneError(ValueError):
    """IANA 时区不存在。"""


@dataclass(frozen=True)
class DstIssue:
    kind: str  # "ambiguous"（重复）| "nonexistent"（不存在）
    field: str
    raw: str
    tz: str
    utc_fold0: datetime
    utc_fold1: datetime
    suggestions: list[str]


def get_zone(tzname: str) -> ZoneInfo:
    try:
        return ZoneInfo(tzname)
    except ZoneInfoNotFoundError as exc:
        raise UnknownTimezoneError(f"未知 IANA 时区: {tzname!r}") from exc


def _format_suggestion(dt_utc: datetime) -> str:
    """给出两种可直接粘贴回输入框的修正写法：UTC 绝对时间。"""
    s = dt_utc.astimezone(timezone.utc).isoformat()
    return s


def parse_wall_time(raw: str, tzname: str, field: str) -> tuple[datetime, DstIssue | None]:
    """解析本地墙上时间，返回 (UTC aware datetime, DST 问题)。

    返回的 datetime 统一为 UTC。若存在 DST 重复/不存在问题，datetime 仍会给出一个
    基于 fold=0 的参考值，但调用方应当因 issue 而拒绝该输入。
    """
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise TimeParseError(f"{field}: 无法解析时间 {raw!r}（需要 ISO 8601，如 2027-03-14T02:30）") from exc

    if dt.tzinfo is not None:
        # 已带绝对偏移量：不存在 DST 歧义。
        return dt.astimezone(timezone.utc), None

    tz = get_zone(tzname)
    d0 = dt.replace(tzinfo=tz, fold=0)
    d1 = dt.replace(tzinfo=tz, fold=1)
    u0 = d0.astimezone(timezone.utc)
    u1 = d1.astimezone(timezone.utc)

    if u0 == u1:
        return u0, None

    off0 = d0.utcoffset()
    off1 = d1.utcoffset()
    assert off0 is not None and off1 is not None

    if off0 > off1:
        # 秋令时回拨：墙上时间重复两次。fold=0 是第一次（偏移更大），fold=1 是第二次。
        local0 = u0.astimezone(tz).replace(tzinfo=None).isoformat()
        local1 = u1.astimezone(tz).replace(tzinfo=None).isoformat()
        issue = DstIssue(
            kind="ambiguous",
            field=field,
            raw=raw,
            tz=tzname,
            utc_fold0=u0,
            utc_fold1=u1,
            suggestions=[
                f"{_format_suggestion(u0)}  （第一次 {local0}，夏令时）",
                f"{_format_suggestion(u1)}  （第二次 {local1}，标准时）",
            ],
        )
    else:
        # 春令时前拨：墙上时间不存在。最近的两个合法时刻是 gap 的两侧。
        def _off(td) -> str:
            total = int(td.total_seconds())
            sign = "+" if total >= 0 else "-"
            total = abs(total)
            return f"UTC{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"

        issue = DstIssue(
            kind="nonexistent",
            field=field,
            raw=raw,
            tz=tzname,
            utc_fold0=u0,
            utc_fold1=u1,
            suggestions=[
                f"{_format_suggestion(u0)}  （gap 前，偏移 {_off(off0)}）",
                f"{_format_suggestion(u1)}  （gap 后，偏移 {_off(off1)}）",
            ],
        )
    return u0, issue


def to_local_display(dt_utc: datetime, tzname: str) -> str:
    """把 UTC 时刻格式化为某时区的本地时间（含偏移量）。"""
    tz = get_zone(tzname)
    return dt_utc.astimezone(tz).isoformat()
