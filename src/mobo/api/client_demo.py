"""调用方示例：生成样本并把请求序列化到 request.json。"""

import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
DEMO_DIR = PROJECT_DIR / "temp" / "api_demo"
REQUEST_ID = "req_demo_002"  # 端上生成并保证唯一；重复使用该 ID 可直接获取已有结果。
REQUEST_DIR = DEMO_DIR / "requests" / REQUEST_ID
REQUEST_FILE = REQUEST_DIR / "request.json"
VARIABLES = ["workpiece_temperature", "die_temperature", "speed", "grain", "load"]


def generate_dataset(path: Path, count: int = 120, seed: int = 42) -> None:
    """调用方生成三输入、两输出的模拟样本。"""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(875, 965, count)
    x2 = rng.uniform(300, 700, count)
    x3 = rng.uniform(10, 50, count)
    a, b, c = (x1 - 875) / 90, (x2 - 300) / 400, (x3 - 10) / 40
    grain = 18 + 7 * (1 - a) ** 2 + 4 * b + 2 * (c - 0.4) ** 2
    load = 250_000 + 70_000 * a + 30_000 * (1 - b) ** 2 + 40_000 * c**2
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([x1, x2, x3, grain, load]), delimiter="\t")


def build_request(dataset: Path) -> dict:
    """构造完全由调用方提供的业务参数。"""
    return {
        "request_id": REQUEST_ID,
        "data_file": str(dataset),
        "all_var_list": VARIABLES,
        "input_var_count": 3,
        "models": [
            {"model_index": 0, "params": {"degree": 2}},
            {"model_index": 1, "params": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}},
            {"model_index": 2, "params": {"n_estimators": 300, "n_jobs": -1}},
            {"model_index": 3, "params": {"alpha": 0.1, "n_restarts_optimizer": 20}},
        ],
        "evaluation": {"n_splits": 3},
        "optimization": {
            "objective_names": ["grain", "load"],
            "decision_var_indices": [0, 1, 2],
            "decision_var_names": VARIABLES[:3],
            "decision_bounds": [
                {"lower": 875, "upper": 965},
                {"lower": 300, "upper": 700},
                {"lower": 10, "upper": 50},
            ],
            "constraints": [
                {"target_obj": "grain", "constraint_kind": "upper", "limit_value": 26},
                {"target_obj": "load", "constraint_kind": "upper", "limit_value": 340_000},
            ],
            "objective_config": [
                {"name": "grain", "minimize": True},
                {"name": "load", "minimize": True},
            ],
            "optimizer_config": {
                "pop_size": 40,
                "n_offsprings": 40,
                "eliminate_duplicates": True,
                "n_gen": 30,
                "seed": 42,
            },
        },
    }


def main() -> None:
    dataset = REQUEST_DIR / "samples.tsv"
    generate_dataset(dataset)
    serialized_request = json.dumps(build_request(dataset), ensure_ascii=False, indent=2)
    REQUEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    REQUEST_FILE.write_text(serialized_request, encoding="utf-8")
    print(f"请求已生成：{REQUEST_FILE}")


if __name__ == "__main__":
    main()
