"""服务层 (:mod:`mobo.automation.service`) 测试。

打桩 :class:`ForgingTask` 与采样，并把任务状态目录重定向到 ``tmp_path``，
验证服务接口的返回结构、state.json 落盘与「仅凭 task_id 续跑」的行为。
"""

import pytest

from mobo.automation import service
from mobo.common import task_store


@pytest.fixture(autouse=True)
def _tmp_tasks(monkeypatch, tmp_path):
    """把 task_store 的任务目录重定向到 tmp_path，避免污染仓库 data/。"""
    monkeypatch.setattr(task_store, "task_dir", lambda tid: tmp_path / "tasks" / tid)


class _FakeTask:
    """记录各阶段调用的假 ForgingTask。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.key_files = ["k0.KEY"]
        self.db_files = ["d0.DB"]
        self.result_txt_dir = "res_txt"

    def generate_keys(self):
        self.calls.append("generate_keys")

    def load_samples_into_table(self):
        self.calls.append("load_samples_into_table")

    def prepare_db_files(self):
        self.calls.append("prepare_db_files")
        return []

    def run_solver(self):
        self.calls.append("run_solver")

    def extract(self):
        self.calls.append("extract")


def _paths(tmp_path):
    return {
        "smp_file": str(tmp_path / "smp.txt"),
        "std_key_file": str(tmp_path / "MODEL.KEY"),
        "temp_key_path": str(tmp_path / "temp_key"),
        "res_db_path": str(tmp_path / "res_db"),
        "res_key_path": str(tmp_path / "res_key"),
        "res_txt_path": str(tmp_path / "res_txt"),
        "process_info_file": str(tmp_path / "process_info.json"),
    }


def test_create_sampling_task_zero_returns_empty():
    assert service.create_sampling_task("t1", "/tmp", "lhs", {"a": (0.0, 1.0)}, 0) == {}


def test_create_sampling_task_success(monkeypatch):
    monkeypatch.setattr(service, "generate_sample_file", lambda *a, **k: "/tmp/INlhs.txt")
    res = service.create_sampling_task("t1", "/tmp", "lhs", {"a": (0.0, 1.0)}, 10)
    assert res["status"] == "success"
    # 抽样结果落盘
    state = task_store.load("t1")
    assert state["data"]["sample_file"] == "/tmp/INlhs.txt"


def test_create_sampling_task_failure(monkeypatch):
    def boom(*a, **k):
        raise ValueError("bad")

    monkeypatch.setattr(service, "generate_sample_file", boom)
    res = service.create_sampling_task("t1", "/tmp", "lhs", {"a": (0.0, 1.0)}, 10)
    assert res["status"] == "failed"


def test_init_execution_task_missing_paths():
    res = service.init_execution_task("t1", {}, [["temp"]], [["grain"]], [False], 10)
    assert res["status"] == "failed"


def test_init_execution_task_persists_req(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "ForgingTask", _FakeTask)
    res = service.init_execution_task(
        "t1", _paths(tmp_path), [["temp"], ["workpiece"]], [["grain"], ["workpiece"]], [False], 100
    )
    assert res["status"] == "success"
    # 输入参数已落盘，供后续步骤续跑
    state = task_store.load("t1")
    assert state["req"]["max_step"] == 100
    assert state["req"]["param_table"] == [["temp"], ["workpiece"]]
    assert state["stage"] == "generate_keys" and state["status"] == "finished"


def test_run_execution_step_unknown_task():
    assert service.run_execution_step("nope")["status"] == "failed"


def test_run_and_extract_resume_from_disk(monkeypatch, tmp_path):
    """init 落盘后，run/extract 仅凭 task_id 从磁盘重建任务续跑。"""
    monkeypatch.setattr(service, "ForgingTask", _FakeTask)
    service.init_execution_task(
        "t1", _paths(tmp_path), [["temp"], ["workpiece"]], [["grain"], ["workpiece"]], [False], 100
    )

    # 只传 task_id，不再传任何参数
    res = service.run_execution_step("t1")
    assert res["status"] == "success"
    assert task_store.load("t1")["stage"] == "run_solver"

    res = service.run_extract_data("t1")
    assert res["status"] == "success"
    assert task_store.load("t1")["stage"] == "extract"


def test_query_execution_status_unknown():
    assert service.query_execution_status("nope")["status"] == "failed"


def test_query_execution_status_reads_state(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "ForgingTask", _FakeTask)
    service.init_execution_task(
        "t1", _paths(tmp_path), [["temp"], ["workpiece"]], [["grain"], ["workpiece"]], [False], 100
    )
    res = service.query_execution_status("t1")
    assert res["status"] == "finished"
    assert "generate_keys" in res["message"]


def test_state_history_accumulates(monkeypatch, tmp_path):
    """完整记录各阶段的状态转移，而非覆盖。"""
    monkeypatch.setattr(service, "ForgingTask", _FakeTask)
    service.init_execution_task(
        "t1", _paths(tmp_path), [["temp"], ["workpiece"]], [["grain"], ["workpiece"]], [False], 100
    )
    service.run_execution_step("t1")
    service.run_extract_data("t1")
    stages = [h["stage"] for h in task_store.load("t1")["history"]]
    assert stages == ["init", "generate_keys", "run_solver", "extract"]


def test_run_without_record_needs_params(monkeypatch, tmp_path):
    """无任务记录且不传参 -> 报错；补齐传参 -> 成功并落盘。"""
    monkeypatch.setattr(service, "ForgingTask", _FakeTask)
    # 既无记录也无传入参数
    assert service.run_execution_step("fresh")["status"] == "failed"

    # 传入参数补齐则可续跑，并回填记录
    res = service.run_execution_step(
        "fresh",
        paths_config=_paths(tmp_path),
        param_table=[["temp"], ["workpiece"]],
        target_table=[["grain"], ["workpiece"]],
        in_progress=[False],
        max_step=100,
    )
    assert res["status"] == "success"
    assert task_store.load("fresh")["req"]["max_step"] == 100


def test_record_takes_precedence_over_overrides(monkeypatch, tmp_path):
    """已有记录时，传入的 overrides 不覆盖记录里的参数。"""
    monkeypatch.setattr(service, "ForgingTask", _FakeTask)
    service.init_execution_task(
        "t1", _paths(tmp_path), [["temp"], ["workpiece"]], [["grain"], ["workpiece"]], [False], 100
    )
    service.run_execution_step("t1", max_step=999)
    assert task_store.load("t1")["req"]["max_step"] == 100


def test_process_info_file_passed_to_task(monkeypatch, tmp_path):
    """paths_config 里的 process_info_file 会透传给 ForgingTask（供求解续跑）。"""
    captured = {}

    class _Capture(_FakeTask):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)

    monkeypatch.setattr(service, "ForgingTask", _Capture)
    service.init_execution_task(
        "t1", _paths(tmp_path), [["temp"], ["workpiece"]], [["grain"], ["workpiece"]], [False], 100
    )
    assert captured["process_info_file"] == str(tmp_path / "process_info.json")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("MODEL0.KEY", 0),
        ("MODEL2.KEY", 2),
        ("MODEL10.KEY", 10),
        ("MODEL199.KEY", 199),
    ],
)
def test_key_sample_index_parses_numeric_suffix(name, expected):
    assert service._key_sample_index(f"/tmp/temp_key/{name}", "MODEL") == expected


def test_key_sample_index_handles_template_stem_with_digits():
    """模板名本身带数字时，仍能正确剥前缀取样本序号。"""
    assert service._key_sample_index("/x/CASE202410.KEY", "CASE2024") == 10


def test_key_sample_index_unparsable_sorts_last():
    """无法解析序号的文件名排到末尾（返回极大值），不打乱正常样本顺序。"""
    assert service._key_sample_index("/x/weird.KEY", "MODEL") == 2 ** 63 - 1


def test_rebuild_task_restores_key_files_in_sample_order(monkeypatch, tmp_path):
    """续跑重建时按样本序号数值排序恢复 key_files（≥11 个样本，字典序会错乱）。"""
    monkeypatch.setattr(service, "ForgingTask", _FakeTask)
    paths = _paths(tmp_path)
    # 造 12 个乱序落盘的 KEY 文件（含两位数序号，字典序会把 10/11 排到 2 之前）
    temp_key = tmp_path / "temp_key"
    temp_key.mkdir()
    for i in range(12):
        (temp_key / f"MODEL{i}.KEY").write_text("")
    service.init_execution_task(
        "t1", paths, [["temp"], ["workpiece"]], [["grain"], ["workpiece"]], [False], 100
    )

    task = service._rebuild_task("t1")
    # 列表位置 i 必须恒等于文件名里的样本序号
    stems = [p.rsplit("/", 1)[-1] for p in task.key_files]
    assert stems == [f"MODEL{i}.KEY" for i in range(12)]


def _setup_misaligned_res_db(tmp_path, n=12):
    """造 n 个 KEY + 按旧字典序错位落盘的 res_db 目录，返回 {真实样本号: DB 名}。"""
    paths = _paths(tmp_path)
    temp_key = tmp_path / "temp_key"
    temp_key.mkdir()
    for i in range(n):
        (temp_key / f"MODEL{i}.KEY").write_text("")
    # 旧代码用字典序 enumerate 生成目录号：MODEL0,MODEL1,MODEL10,MODEL11,MODEL2,...
    res_db = tmp_path / "res_db"
    res_db.mkdir()
    old_order = sorted(f"MODEL{i}" for i in range(n))
    for dir_idx, stem in enumerate(old_order):
        d = res_db / str(dir_idx)
        d.mkdir()
        (d / f"{stem}.DB").write_text(stem)  # 内容=真实样本 stem，用于校验不丢
    service.init_execution_task(
        "t1", paths, [["temp"], ["workpiece"]], [["grain"], ["workpiece"]], [False], 100
    )
    return res_db


def test_align_result_db_dirs_fixes_misordered(tmp_path):
    """对齐后 res_db/<i>/ 的目录号恒等于其内 DB 的真实样本号，内容不丢。"""
    res_db = _setup_misaligned_res_db(tmp_path, 12)
    res = service.align_result_db_dirs("t1")
    assert res["status"] == "success"

    for d in res_db.iterdir():
        dbs = [f for f in d.iterdir() if f.suffix == ".DB"]
        assert len(dbs) == 1
        stem = dbs[0].stem  # 形如 MODEL7
        real_idx = int(stem.replace("MODEL", ""))
        assert int(d.name) == real_idx  # 目录号 == 真实样本号
        assert dbs[0].read_text() == stem  # 内容完整
    # 无临时残留
    assert not any(x.name.startswith(".__align_tmp") for x in res_db.iterdir())


def test_align_result_db_dirs_idempotent_and_dry_run(tmp_path):
    """预演不改盘；对齐后再跑显示无需改动（幂等）。"""
    _setup_misaligned_res_db(tmp_path, 12)
    # 预演：报告有目录待改，但不动磁盘
    dry = service.align_result_db_dirs("t1", apply=False)
    assert dry["status"] == "success" and "预演" in dry["message"]

    service.align_result_db_dirs("t1")  # 真正对齐
    again = service.align_result_db_dirs("t1")  # 再跑
    assert again["status"] == "success" and "无需改动" in again["message"]


def test_align_result_db_dirs_partial(tmp_path):
    """只有部分 DB 目录存在（求解未完成）时也能正确对齐、跳过缺失项。"""
    paths = _paths(tmp_path)
    temp_key = tmp_path / "temp_key"
    temp_key.mkdir()
    for i in range(12):
        (temp_key / f"MODEL{i}.KEY").write_text("")
    res_db = tmp_path / "res_db"
    res_db.mkdir()
    # 仅前 5 个字典序目录已求解完成
    old_order = sorted(f"MODEL{i}" for i in range(12))
    for dir_idx, stem in enumerate(old_order[:5]):
        d = res_db / str(dir_idx)
        d.mkdir()
        (d / f"{stem}.DB").write_text(stem)
    service.init_execution_task(
        "t1", paths, [["temp"], ["workpiece"]], [["grain"], ["workpiece"]], [False], 100
    )

    res = service.align_result_db_dirs("t1")
    assert res["status"] == "success"
    for d in res_db.iterdir():
        stem = next(f for f in d.iterdir() if f.suffix == ".DB").stem
        assert int(d.name) == int(stem.replace("MODEL", ""))


def test_run_extract_data_resume_loads_samples_and_no_index_col(monkeypatch, tmp_path):
    """extract 续跑：param_table 仅 2 行表头也能从样本文件补回样本行，
    产出数据集无行号列，且参数列与目标列按样本顺序正确配对（含两位数序号）。"""
    from mobo.automation import extract
    from mobo.automation.config import DeformConfig

    # 打桩 DEFORM 相关子过程，避免依赖真实环境
    monkeypatch.setattr(extract, "db_to_key",
                        lambda db, key, step: open(key, "w", encoding="utf-8").close())
    monkeypatch.setattr(extract, "read_key_frames", lambda files: [["frame"]])
    # 提取器回显它收到的样本第一个参数值，便于校验参数-目标配对
    monkeypatch.setattr(DeformConfig, "get_target_function",
                        classmethod(lambda cls, name: lambda kf, fr, obj, prog, sc: "T"))

    n = 12
    paths = _paths(tmp_path)
    (tmp_path / "MODEL.KEY").write_text("")  # 空模板，供 init 的 generate_keys 读取
    # 样本文件：第 i 行首列 = 样本序号 i（用于校验 param_table[i+2] 对齐）
    sample_lines = [f"{i} 30\n" for i in range(n)]
    (tmp_path / "smp.txt").write_text("".join(sample_lines))
    # temp_key：模板名 MODEL，样本序号 0..11（含两位数，字典序会乱）
    temp_key = tmp_path / "temp_key"; temp_key.mkdir()
    for i in range(n):
        (temp_key / f"MODEL{i}.KEY").write_text("")
    # res_db：已按数值序对齐的目录（<i>/MODEL<i>.DB），模拟求解完成
    res_db = tmp_path / "res_db"; res_db.mkdir()
    for i in range(n):
        d = res_db / str(i); d.mkdir()
        (d / f"MODEL{i}.DB").write_text("")

    service.init_execution_task(
        "t1", paths, [["temp"], ["workpiece"]], [["grain"], ["workpiece"], [1]], [False], 1
    )
    # 记录里 param_table 仍只有 2 行（样本行未落盘）
    assert task_store.load("t1")["req"]["param_table"] == [["temp"], ["workpiece"]]

    res = service.run_extract_data("t1")
    assert res["status"] == "success"  # 不再 IndexError

    # 读产出数据集，校验无行号列且首列 == 样本序号（顺序正确）
    out_files = list((tmp_path / "res_txt").glob("*_result.txt"))
    assert len(out_files) == 1
    rows = [ln.split("\t") for ln in out_files[0].read_text().splitlines()]
    assert len(rows) == n
    for i, row in enumerate(rows):
        assert row[0] == str(i)   # 首列是样本首参数(=序号 i)，非行号
        assert row[-1] == "T"     # 末列是目标值




