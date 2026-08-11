"""结果数据集提取 (:mod:`mobo.automation.extract`) 测试。

打桩 ``db_to_key`` 与提取函数，避免依赖真实 DEFORM，验证编排逻辑：逐步导出、
按目标调用提取器、汇总写出数据集文件。
"""

import os

from mobo.automation import extract
from mobo.automation.config import DeformConfig


def test_export_all_steps(monkeypatch, tmp_path):
    exported = []

    def fake_db_to_key(db, key, step):
        exported.append((db, key, step))
        # 模拟 DEFORM 真正生成了文件，使 while 循环退出
        open(key, "w", encoding="utf-8").close()

    monkeypatch.setattr(extract, "db_to_key", fake_db_to_key)
    keys = extract._export_all_steps(str(tmp_path / "model.DB"), str(tmp_path / "out"), 3)
    assert len(keys) == 3
    assert all(os.path.exists(k) for k in keys)
    assert [s for _, _, s in exported] == ["0", "1", "2"]


def test_extract_dataset_writes_result(monkeypatch, tmp_path):
    # db_to_key 直接落一个空 KEY 文件
    monkeypatch.setattr(
        extract, "db_to_key",
        lambda db, key, step: open(key, "w", encoding="utf-8").close(),
    )
    # read_key_frames 返回占位帧
    monkeypatch.setattr(extract, "read_key_frames", lambda files: [["frame"]])
    # 目标提取函数：记录收到的统一签名参数，返回固定值
    received = []

    def fake_extractor(key_files, frames, obj, prog, select_component):
        received.append((obj, frames, select_component))
        return "42.00"

    monkeypatch.setattr(DeformConfig, "get_target_function", classmethod(lambda cls, name: fake_extractor))

    param_table = [
        ["temp", "speed"],
        ["workpiece", "topdie"],
        ["900", "30"],  # 样本 0
    ]
    target_table = [["grain"], ["workpiece"], [3]]
    db_files = [str(tmp_path / "res" / "model.DB")]
    os.makedirs(os.path.dirname(db_files[0]), exist_ok=True)

    out = extract.extract_dataset(
        db_files,
        str(tmp_path / "keys"),
        1,
        param_table,
        target_table,
        [False],
        str(tmp_path / "result"),
    )
    assert out.endswith("_result.txt")
    content = open(out, encoding="utf-8").read()
    # 工艺参数 + 目标值（无行号列）
    assert "900" in content and "30" in content and "42.00" in content
    # 不再写行号列：首行应直接以参数值开头
    assert content.splitlines()[0].split("\t")[0] == "900"
    # 提取函数应收到对象 ID（"1"）与 select_component（3）
    assert received == [("1", [["frame"]], 3)]


def test_extract_dataset_routes_roundness(monkeypatch, tmp_path):
    """碾环圆度目标通过统一提取流程调用（对象名转 ID，用 KEY 文件计算）。"""
    monkeypatch.setattr(
        extract, "db_to_key",
        lambda db, key, step: open(key, "w", encoding="utf-8").close(),
    )
    monkeypatch.setattr(extract, "read_key_frames", lambda files: [["frame"]])

    import mobo.automation.config as config
    calls = []
    monkeypatch.setattr(
        config, "extract_ring_roundness",
        lambda key_path, *, which, object_id: calls.append((key_path, which, object_id)) or 1.5,
    )

    param_table = [["temp"], ["workpiece"], ["900"]]
    target_table = [["roundness_inner"], ["workpiece"], [None]]
    db_files = [str(tmp_path / "res" / "model.DB")]
    os.makedirs(os.path.dirname(db_files[0]), exist_ok=True)

    out = extract.extract_dataset(
        db_files, str(tmp_path / "keys"), 2, param_table, target_table, [False],
        str(tmp_path / "result"),
    )
    content = open(out, encoding="utf-8").read()
    assert "1.500000" in content  # 圆度 6 位小数
    # 用最终步 KEY 文件、inner 圈、对象 ID=1
    assert len(calls) == 1
    key_path, which, object_id = calls[0]
    assert which == "inner" and object_id == 1
    assert key_path.endswith("model1.KEY")  # max_step=2 -> 最终步索引 1
