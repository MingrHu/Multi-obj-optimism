"""Qt-independent helpers for the MOBO desktop interface."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


TERMINAL_STATES = frozenset({"finished", "failed", "stopped", "not_started"})
RUNNING_STATES = frozenset({"queued", "running", "stopping"})


class ApiError(RuntimeError):
    """A backend transport or business-protocol failure."""


@dataclass(frozen=True)
class StatusView:
    """Presentation-friendly interpretation of a backend task status."""

    text: str
    tone: str
    active: bool


@dataclass(frozen=True)
class CleanTable:
    """Finite numeric data prepared for model training."""

    headers: list[str]
    rows: list[list[float]]
    dropped_columns: list[str]
    dropped_row_count: int


STATUS_VIEWS = {
    "not_started": StatusView("未开始", "neutral", False),
    "queued": StatusView("排队中", "info", True),
    "running": StatusView("运行中", "info", True),
    "stopping": StatusView("正在停止", "warning", True),
    "stopped": StatusView("已停止", "warning", False),
    "finished": StatusView("已完成", "success", False),
    "failed": StatusView("失败", "danger", False),
}

STAGE_TEXT = {
    "not_started": "未开始",
    "queued": "等待调度",
    "training": "训练模型",
    "evaluating": "交叉验证",
    "optimization": "准备优化",
    "optimizing": "执行优化",
    "stopping": "正在停止",
    "stopped": "已停止",
    "finished": "已完成",
    "failed": "失败",
}


def status_view(status: str | None) -> StatusView:
    """Return a stable label/tone for known and future backend states."""
    normalized = str(status or "not_started").lower()
    return STATUS_VIEWS.get(normalized, StatusView(normalized, "neutral", False))


def stage_text(stage: Any) -> str:
    """Return a concise Chinese label for backend workflow stages."""
    normalized = str(stage or "not_started").strip().lower()
    return STAGE_TEXT.get(normalized, normalized or "—")


def format_timestamp(value: Any, local_timezone: tzinfo | None = None) -> str:
    """Format an API timestamp in the workstation's local timezone."""
    text = str(value or "").strip()
    if not text or text == "—":
        return "—"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(local_timezone)
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def rows_by_fields(values: dict[str, list[Any]], fields: Iterable[str]) -> list[list[Any]]:
    """Transpose the API's column-oriented payload into rows."""
    names = list(fields)
    if not names:
        return []
    lengths = {len(values.get(name, [])) for name in names}
    if len(lengths) > 1:
        raise ValueError("字段数据长度不一致")
    return [[values[name][index] for name in names] for index in range(next(iter(lengths), 0))]


def columns_from_rows(headers: list[str], rows: list[list[Any]]) -> dict[str, list[Any]]:
    """Convert table rows into the field-oriented shape used by chart controls."""
    return {
        name: [row[index] if index < len(row) else None for row in rows]
        for index, name in enumerate(headers)
    }


def finite_float(value: Any) -> float | None:
    """Return a finite float, mapping common boolean text and rejecting NaN/Inf."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return 1.0 if value.strip().lower() == "true" else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def numeric_field_names(headers: list[str], rows: list[list[Any]]) -> list[str]:
    """Return fields containing at least one finite numeric value."""
    result = []
    for index, name in enumerate(headers):
        if any(index < len(row) and finite_float(row[index]) is not None for row in rows):
            result.append(name)
    return result


def prepare_numeric_table(headers: list[str], rows: list[list[Any]]) -> CleanTable:
    """Drop empty/non-numeric columns and incomplete rows before model training."""
    usable_names = set(numeric_field_names(headers, rows))
    usable_indices = [index for index, name in enumerate(headers) if name in usable_names]
    dropped_columns = [name for name in headers if name not in usable_names]
    clean_rows: list[list[float]] = []
    dropped_row_count = 0
    for row in rows:
        values = [finite_float(row[index]) if index < len(row) else None for index in usable_indices]
        if any(value is None for value in values):
            dropped_row_count += 1
            continue
        clean_rows.append([float(value) for value in values if value is not None])
    return CleanTable(
        headers=[headers[index] for index in usable_indices],
        rows=clean_rows,
        dropped_columns=dropped_columns,
        dropped_row_count=dropped_row_count,
    )


def read_tabular_file(path: str | Path) -> tuple[list[str], list[list[Any]]]:
    """Read CSV/TSV data and infer a header when the file has none."""
    file_path = Path(path)
    raw_lines = [line for line in file_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not raw_lines:
        return [], []
    delimiter = "\t" if "\t" in raw_lines[0] else ","
    cells = [[cell.strip() for cell in line.split(delimiter)] for line in raw_lines]

    def parse(value: str) -> Any:
        try:
            return float(value)
        except ValueError:
            return value

    parsed = [[parse(cell) for cell in row] for row in cells]
    first_is_header = any(not isinstance(value, (int, float)) for value in parsed[0])
    width = max(len(row) for row in parsed)
    headers = [str(value) for value in parsed[0]] if first_is_header else [f"字段 {i + 1}" for i in range(width)]
    return headers, parsed[1:] if first_is_header else parsed


class ApiClient:
    """Small synchronous client; callers run it in a Qt worker thread."""

    def __init__(self, base_url: str = "http://127.0.0.1:5000", timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        params: list[tuple[str, Any]] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urlencode(params, doseq=True)}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = Request(url, data=body, method=method.upper())
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", "application/json; charset=utf-8")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                message = json.loads(exc.read().decode("utf-8")).get("message", str(exc))
            except (ValueError, UnicodeDecodeError):
                message = str(exc)
            raise ApiError(message) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ApiError(f"无法连接后端服务：{exc}") from exc
        except (ValueError, UnicodeDecodeError) as exc:
            raise ApiError("后端返回了无法解析的数据") from exc
        if not isinstance(result, dict) or result.get("code") != 0:
            raise ApiError(str(result.get("message", "后端请求失败")) if isinstance(result, dict) else "后端请求失败")
        data = result.get("data", {})
        return data if isinstance(data, dict) else {"value": data}

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/health")

    def list_doe(self) -> dict[str, Any]:
        return self.request("GET", "/api/v1/doe/list")

    def add_doe(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/api/v1/doe/add", payload=payload)

    def generate_sample(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/api/v1/hust/doe/sample/generate", payload=payload)

    def generate_training_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/api/v1/hust/doe/dataset/generate", payload=payload)

    def save_training_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/api/v1/hust/doe/dataset/save", payload=payload)

    def get_data(self, doe_id: str, resource_id: str, fields: list[str]) -> dict[str, Any]:
        params = [("id", doe_id), ("resource_id", resource_id)] + [("fields", field) for field in fields]
        return self.request("GET", "/api/v1/hust/doe/data/get", params=params)

    def start_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/api/v1/hust/doe/train/startTrain", payload=payload)

    def training_progress(self, doe_id: str) -> dict[str, Any]:
        return self.request("GET", "/api/v1/hust/doe/train/progress", params={"id": doe_id})

    def stop_training(self, doe_id: str) -> dict[str, Any]:
        return self.request("POST", "/api/v1/hust/doe/train/stop", payload={"id": doe_id})

    def start_optimization(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/api/v1/hust/doe/optimize/start", payload=payload)

    def optimization_progress(self, doe_id: str) -> dict[str, Any]:
        return self.request("GET", "/api/v1/hust/doe/optimize/getById", params={"id": doe_id})

    def stop_optimization(self, doe_id: str) -> dict[str, Any]:
        return self.request("POST", "/api/v1/hust/doe/optimize/stop", payload={"id": doe_id})
