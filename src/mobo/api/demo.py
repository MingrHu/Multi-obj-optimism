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


def rows_by_fields(data: dict, fields: list[str]) -> list[list[float]]:
    # 按字段顺序把接口返回的列式数据转换为训练接口需要的二维样本
    return [list(row) for row in zip(*(data[field] for field in fields), strict=True)]


def main() -> None:
    # 演示完整 DOE 生命周期 创建数据 训练评价 推理 优化 查询结果
    doe_id = f"demo_doe_{int(time.time())}"
    input_names = ["temperature", "speed"]
    target_names = ["grain", "load"]
    param_ranges = {"temperature": [900, 1100], "speed": [10, 50]}

    # 1 创建 DOE 任务
    call("POST", "/api/v1/doe/add", json={
        "id": doe_id,
        "name": "HTTP DOE 演示",
        "metadata": {
            "workpiece": "ring"
        },
    })

    # 2 生成 LHS 样本
    sample = call("POST", "/api/v1/hust/doe/sample/generate", json={
        "id": doe_id,
        "method": "lhs",
        "param_ranges": param_ranges,
        "n_samples": 12,
    })
    if sample["data"]["sample_count"] < sample["data"]["n_samples"]:
        raise RuntimeError(f"样本数量异常 {sample['data']}")

    # 3 通过 GET 接口只获取样本中的温度字段
    call("GET", "/api/v1/hust/doe/data/get", params=[
        ("id", doe_id),
        ("resource_id", sample["data"]["resource_id"]),
        ("fields", "temperature"),
    ])

    # 4 生成测试训练数据集
    generated_dataset = call("POST", "/api/v1/hust/doe/dataset/generate", json={
        "id": doe_id,
        "input_names": input_names,
        "target_names": target_names,
        "param_ranges": param_ranges,
        "n_samples": 80,
        "seed": 42,
    })

    # 5 按字段获取训练数据集
    dataset = call("GET", "/api/v1/hust/doe/data/get", params=[
        ("id", doe_id),
        ("resource_id", generated_dataset["data"]["resource_id"]),
        ("fields", "temperature"),
        ("fields", "speed"),
        ("fields", "grain"),
        ("fields", "load"),
    ])["data"]["values"]

    # 6 训练代理模型并评价
    call("POST", "/api/v1/hust/doe/train/startTrain", json={
        "id": doe_id,
        "data_source": {
            "input_data": {
                "labels": input_names,
                "samples": rows_by_fields(dataset, input_names),
            },
            "output_data": {
                "labels": target_names,
                "samples": rows_by_fields(dataset, target_names),
            },
        },
        "models": [{"name": "RF", "params": {"n_estimators": 300, "n_jobs": -1}}],
        "evaluation": {
            "enabled": True,
            "method": "k_fold",
            "n_splits": 3,
            "random_state": 42
        },
    })

    # 7 等待训练完成
    training = wait_for_terminal("/api/v1/hust/doe/train/progress", doe_id)
    if training["data"]["status"] != "finished":
        raise RuntimeError(f"代理模型训练未完成 {training['data']}")

    # 8 推理测试 只返回 grain
    inference = call("POST", "/api/v1/hust/doe/inference/startInference", json={
        "id": doe_id,
        "inputs": {
            "temperature": [1000],
            "speed": [30]
        },
        "fields": ["grain"],
    })

    # 9 通过统一 GET 接口获取最近一次推理的 load
    call("GET", "/api/v1/hust/doe/data/get", params=[
        ("id", doe_id),
        ("resource_id", inference["data"]["resource_id"]),
        ("fields", "load"),
    ])

    # 10 优化任务
    call("POST", "/api/v1/hust/doe/optimize/start", json={
        "id": doe_id,
        "mode": "multi",
        "objectives": [
            {"name": "grain", "direction": "min"},
            {"name": "load", "direction": "min"},
        ],
        "constraints": [],
        "decision_variables": [
            {"name": "temperature", "lower": 900, "upper": 1100},
            {"name": "speed", "lower": 10, "upper": 50},
        ],
        "algorithm": {
            "name": "nsga2",
            "params": {
                "pop_size": 20,
                "n_offsprings": 10,
                "eliminate_duplicates": True,
                "n_gen": 10,
                "seed": 42,
            },
        },
    })

    # 11 等待优化完成
    optimization = wait_for_terminal(
        "/api/v1/hust/doe/optimize/getById", doe_id
    )

    # 12 检查优化结果
    if optimization["data"]["status"] != "finished":
        raise RuntimeError(f"优化任务未完成 {optimization['data']}")

    # 13 按字段获取优化结果 文件本身无表头
    call("GET", "/api/v1/hust/doe/data/get", params=[
        ("id", doe_id),
        ("resource_id", optimization["data"]["result"]["resource_id"]),
        ("fields", "temperature"),
        ("fields", "grain"),
        ("fields", "feasible"),
    ])


if __name__ == "__main__":
    main()
