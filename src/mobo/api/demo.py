"""真实 HTTP 请求风格的 DOE API 客户端示例。先运行 ``mobo-api``。"""

from __future__ import annotations

import json
import os
import time

import requests

BASE_URL = os.environ.get("MOBO_API_URL", "http://127.0.0.1:5000")


def call(method: str, path: str, **kwargs):
    # 统一打印 HTTP 状态和 JSON 内容 便于与实际调用方日志直接对照
    response = requests.request(method, BASE_URL + path, timeout=30, **kwargs)
    print(f"{method} {path} -> HTTP {response.status_code}")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    return response.json()


def wait_for_terminal(path: str, doe_id: str, timeout: int = 600):
    # 训练和优化均为后台任务 demo通过查询接口等待最终状态
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = call("GET", path, params={"id": doe_id})
        status = response["data"]["status"]
        if status in {"finished", "failed", "stopped"}:
            return response
        time.sleep(2)
    raise TimeoutError(f"等待任务完成超时 {doe_id}")


def main() -> None:
    # 演示完整 DOE 生命周期 创建数据 训练评价 推理 优化 查询结果
    doe_id = f"demo_doe_{int(time.time())}"
    input_names = ["temperature", "speed"]
    target_names = ["grain", "load"]
    param_ranges = {"temperature": [900, 1100], "speed": [10, 50]}
    call("POST", "/api/v1/doe/add", json={
        "id": doe_id, "name": "HTTP DOE 演示", "metadata": {"workpiece": "ring"},
    })
    dataset = call("POST", "/api/v1/hust/doe/dataset/generate", json={
        "id": doe_id, "input_names": input_names, "target_names": target_names,
        "param_ranges": param_ranges, "n_samples": 80, "seed": 42,
    })["data"]
    call("POST", "/api/v1/hust/doe/train/startTrain", json={
        "id": doe_id, "data_file": dataset["data_file"],
        "all_var_list": dataset["all_var_list"],
        "input_var_count": dataset["input_var_count"],
        "models": [{"model_index": 2}],
        "evaluation": {"enabled": True, "n_splits": 3, "random_state": 42},
    })
    training = wait_for_terminal("/api/hust/v1/doe/train/progress", doe_id)
    if training["data"]["status"] != "finished":
        raise RuntimeError(f"代理模型训练未完成 {training['data']}")

    call("POST", "/api/v1/hust/doe/inference/startInference", json={
        "id": doe_id, "inputs": [[1000, 30]],
    })
    call("POST", "/api/v1/hust/doe/optimize/start", json={
        "id": doe_id, "algorithm": "nsga2", "objective_names": target_names,
        "all_var_list": [*input_names, *target_names], "input_var_count": 2,
        "decision_var_indices": [0, 1], "decision_var_names": input_names,
        "decision_bounds": [
            {"lower": 900, "upper": 1100}, {"lower": 10, "upper": 50},
        ],
        "optimizer_config": {
            "pop_size": 20, "n_offsprings": 10, "eliminate_duplicates": True,
            "n_gen": 10, "seed": 42,
        },
    })
    optimization = wait_for_terminal(
        "/api/v1/hust/doe/optimize/getById", doe_id
    )
    if optimization["data"]["status"] != "finished":
        raise RuntimeError(f"优化任务未完成 {optimization['data']}")


if __name__ == "__main__":
    main()
