"""服务方示例：读取 request.json，完成训练、评价和优化。"""

import json
import os
import sys
from hashlib import sha256
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[3]
DEMO_DIR = PROJECT_DIR / "temp" / "api_demo"
REQUESTS_DIR = DEMO_DIR / "requests"

# 必须在导入 mobo 服务模块前设置，服务产生的模型和任务文件才会进入 temp。
os.environ["MOBO_DATA_DIR"] = str(REQUESTS_DIR)

JsonObject = dict[str, Any]
MODEL_NAMES: dict[int, str] = {0: "PRG", 1: "SVR", 2: "RF", 3: "KM", 4: "DNN"}


def train_models(request: JsonObject) -> dict[str, str]:
    """服务方调用训练接口。"""
    from mobo.api.facade import train_surrogate

    model_ids: dict[str, str] = {}
    for model in request["models"]:
        payload: JsonObject = {
            "data_file": request["data_file"],
            "all_var_list": request["all_var_list"],
            "input_var_count": request["input_var_count"],
            **model,
        }
        response = train_surrogate(json.dumps(payload, ensure_ascii=False))
        if response["code"] != 0:
            raise RuntimeError(response["msg"])
        model_ids[MODEL_NAMES[model["model_index"]]] = response["model_id"]
    return model_ids


def select_best_model(
    request: JsonObject,
    output_dir: Path,
) -> tuple[str, list[JsonObject]]:
    """服务方交叉验证并选择平均评分最高的模型。"""
    from mobo.surrogate.evaluate import SurrogateModelEvaluator

    names = [MODEL_NAMES[item["model_index"]] for item in request["models"]]
    params = {MODEL_NAMES[item["model_index"]]: item["params"] for item in request["models"]}
    evaluator = SurrogateModelEvaluator(
        request["data_file"],
        request["all_var_list"],
        request["input_var_count"],
        n_splits=request["evaluation"]["n_splits"],
        model_params=params,
    )
    summaries = evaluator.evaluate(models=names)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.save_report(
        summaries,
        text_path=str(output_dir / "evaluation.txt"),
        json_path=str(output_dir / "evaluation.json"),
    )
    scores: dict[str, float] = {}
    for name in names:
        model_scores = [
            item.score
            for item in summaries
            if item.model_name == name and item.score is not None
        ]
        scores[name] = mean(model_scores)
    best_model = max(scores.items(), key=lambda item: item[1])[0]
    return best_model, [asdict(item) for item in summaries]


def optimize(request: JsonObject, model_id: str, output_dir: Path) -> JsonObject:
    """服务方补充 model_id 和内部输出路径后调用优化接口。"""
    from mobo.api.facade import run_optimization

    payload: JsonObject = {
        "model_id": model_id,
        "all_var_list": request["all_var_list"],
        "input_var_count": request["input_var_count"],
        **request["optimization"],
        "output_config": {"pareto_txt_path": str(output_dir / "pareto_solutions.tsv")},
    }
    response = run_optimization(json.dumps(payload, ensure_ascii=False))
    if response["code"] != 0:
        raise RuntimeError(response["msg"])
    return response


def run_service(serialized_request: str) -> JsonObject:
    """处理请求；相同 request_id 和请求内容直接返回已有结果。"""
    from mobo.api.validation import validate_task_id

    request = json.loads(serialized_request)
    if not isinstance(request, dict):
        raise ValueError("请求必须是 JSON 对象")
    
    request_id = validate_task_id(request.get("request_id"), "request_id", "req_")
    request_hash = sha256(
        json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    request_dir = REQUESTS_DIR / request_id
    response_file = request_dir / "response.json"

    if response_file.exists():
        response = json.loads(response_file.read_text(encoding="utf-8"))
        if response["request_hash"] != request_hash:
            raise ValueError(f"request_id 已绑定其他请求：{request_id}")
        response["cached"] = True
        return response

    model_ids = train_models(request)
    output_dir = request_dir / "server"
    best_model, evaluation = select_best_model(request, output_dir)
    response = {
        "request_id": request_id,
        "request_hash": request_hash,
        "cached": False,
        "model_ids": model_ids,
        "best_model": best_model,
        "evaluation": evaluation,
        "optimization": optimize(request, model_ids[best_model], output_dir),
    }
    request_dir.mkdir(parents=True, exist_ok=True)
    response_file.write_text(
        json.dumps(response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return response


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    for request_file in sorted(REQUESTS_DIR.glob("*/request.json")):
        response = run_service(request_file.read_text(encoding="utf-8"))
        print(
            f"request_id={response['request_id']}, "
            f"cached={response['cached']}, response={request_file.parent / 'response.json'}"
        )


if __name__ == "__main__":
    main()
