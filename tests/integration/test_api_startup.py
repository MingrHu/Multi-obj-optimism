"""API启动预加载后的并发首次采样集成测试"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from mobo.api import store
from mobo.api.app import create_app


@pytest.mark.integration
def test_preloaded_sampling_handles_concurrent_first_requests(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DOE_TASKS_DIR", tmp_path / "doe_tasks")
    app = create_app({"TESTING": True})
    task_ids = [f"concurrent_sample_{index}" for index in range(4)]
    for task_id in task_ids:
        store.create({"id": task_id})
    barrier = Barrier(len(task_ids))

    def generate(task_id):
        barrier.wait()
        with app.test_client() as client:
            return client.post("/api/v1/hust/doe/sample/generate", json={
                "id": task_id,
                "method": "lhs",
                "param_ranges": {"X1": [0, 1], "X2": [10, 20]},
                "n_samples": 4,
            })

    with ThreadPoolExecutor(max_workers=len(task_ids)) as executor:
        responses = list(executor.map(generate, task_ids))

    assert [response.status_code for response in responses] == [200] * len(task_ids)
    assert [response.json["data"]["sample_count"] for response in responses] == [4] * len(task_ids)
