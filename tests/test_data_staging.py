import json
from pathlib import Path

import pytest

from analyze import data_staging
from analyze.agents import agentic_analyst
from pennsieve_ai_utils.blackboard import Blackboard, DatasetSummary, Hypothesis

INDEX = [{"path": "files/DATA/annotations.tsv", "size": 10, "fileType": "TSV"},
         {"path": "files/DATA/sub-1/a.edf", "size": 100, "fileType": "EDF"},
         {"path": "files/DATA/sub-2/b.edf", "size": 100, "fileType": "EDF"},
         {"path": "files/README.md", "size": 5, "fileType": "Markdown"}]


class FakeFiles:
    dataset_id, version = "530", "3"

    def __init__(self, fail=()):
        self.fail = set(fail)

    def index(self):
        return INDEX

    def download(self, paths, dest):
        for p in paths:
            if p in self.fail:
                raise FileNotFoundError(p)
            (dest / p).parent.mkdir(parents=True, exist_ok=True)
            (dest / p).write_text("x")
        return [dest / p for p in paths]


def board(seed="pennsieve:530"):
    bb = Blackboard("q", seed)
    bb.dataset_summaries = [DatasetSummary(id=seed, title="Seizures", description="", measurements=[],
                                           species="", n_subjects=0, doi="10.1/x")]
    return bb


def hypothesis():
    return Hypothesis("H1", "EDF onset spread", "why", [], "pred", "use the edf files")


def test_staging_writes_the_index_and_a_folder_spread_sample(tmp_path):
    result = data_staging.stage(board(), hypothesis(), tmp_path, files=FakeFiles())
    index = json.loads(result.index_path.read_text())
    assert index["dataset_id"] == "530" and index["file_count"] == 4
    assert [s["path"] for s in result.staged] == [
        "files/DATA/annotations.tsv", "files/DATA/sub-1/a.edf", "files/DATA/sub-2/b.edf"]
    assert (tmp_path / "data/files/DATA/sub-2/b.edf").is_file()
    section = data_staging.prompt_section(result)
    assert "discover_files fetch 530 --version 3 --dest data" in section
    assert "Do not** use AWS credentials" in section


def test_a_dataset_that_is_not_on_pennsieve_stops_staging(tmp_path):
    with pytest.raises(data_staging.DataStagingError, match="not a published Pennsieve dataset"):
        data_staging.stage(board("dandi:000001"), hypothesis(), tmp_path, files=FakeFiles())


def test_an_unreadable_index_stops_staging(tmp_path):
    class Broken(FakeFiles):
        def index(self):
            raise ConnectionError("no route")
    with pytest.raises(data_staging.DataStagingError, match="could not read the file index"):
        data_staging.stage(board(), hypothesis(), tmp_path, files=Broken())


def test_every_sample_download_failing_stops_staging(tmp_path):
    every = {f["path"] for f in INDEX}
    with pytest.raises(data_staging.DataStagingError, match="every sample download failed"):
        data_staging.stage(board(), hypothesis(), tmp_path, files=FakeFiles(fail=every))


def test_one_failed_download_is_recorded_not_fatal(tmp_path):
    bb = board()
    result = data_staging.stage(bb, hypothesis(), tmp_path, files=FakeFiles(fail={"files/DATA/sub-1/a.edf"}))
    assert result.failed == ["files/DATA/sub-1/a.edf"] and len(result.staged) == 2
    assert any("file_error" in str(e) for e in bb.notebook)


def write_status(tmp_path, status):
    (tmp_path / agentic_analyst.STATUS_FILE).write_text(json.dumps(status))


def test_a_tested_status_with_real_files_passes(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/a.fcs").write_text("x")
    write_status(tmp_path, {"status": "tested", "data_files": ["data/a.fcs"]})
    assert agentic_analyst.check_analysis_status(tmp_path)["status"] == "tested"


@pytest.mark.parametrize("status, message", [
    (None, "did not write"),
    ({"status": "blocked", "blocker": "403 from S3", "data_files": []}, "blocked: 403 from S3"),
    ({"status": "tested", "data_files": []}, "lists no data files"),
    ({"status": "tested", "data_files": ["data/missing.fcs"]}, "not in"),
    ({"status": "tested", "data_files": ["../../etc/passwd"]}, "not in"),
    ({"status": "done", "data_files": ["x"]}, "expected 'tested'"),
])
def test_an_analysis_without_data_does_not_pass(tmp_path, status, message):
    if status is not None:
        write_status(tmp_path, status)
    with pytest.raises(RuntimeError, match=message):
        agentic_analyst.check_analysis_status(tmp_path)


def test_the_methods_critic_sees_the_status_report_and_the_no_data_rule(monkeypatch, tmp_path):
    from analyze.agents import methods_critic
    seen = {}

    def fake_call_json(client, **kwargs):
        seen.update(kwargs)
        return {"verdict": "unsound", "issues": [], "suggested_followups": []}

    monkeypatch.setattr(methods_critic, "call_json", fake_call_json)
    import nbformat
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('blocked')", execution_count=1)])
    nbformat.write(nb, tmp_path / "analysis.ipynb")
    write_status(tmp_path, {"status": "blocked", "blocker": "access denied", "data_files": []})
    h = hypothesis()
    h.analysis_summary = "Data could not be downloaded."
    methods_critic.run(Blackboard("q", "s"), client=None, hypothesis=h,
                       notebook_path=tmp_path / "analysis.ipynb")
    assert '"blocker": "access denied"' in seen["user"]
    assert "is never sound" in seen["system"]


def test_a_not_testable_status_is_its_own_outcome(tmp_path):
    write_status(tmp_path, {"status": "not_testable", "reason": "ADCP covers 3 of 12 clones", "data_files": []})
    with pytest.raises(agentic_analyst.NotTestable, match="3 of 12 clones"):
        agentic_analyst.check_analysis_status(tmp_path)


def test_a_critic_issue_without_a_kind_counts_as_fixable(monkeypatch, tmp_path):
    from analyze.agents import methods_critic
    monkeypatch.setattr(methods_critic, "call_json", lambda client, **kw: {
        "verdict": "concerns", "issues": [{"severity": "high", "description": "x"}], "suggested_followups": []})
    import nbformat
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("1", execution_count=1)]),
                   tmp_path / "analysis.ipynb")
    h = hypothesis()
    h.analysis_summary = "s"
    methods_critic.run(Blackboard("q", "s"), client=None, hypothesis=h, notebook_path=tmp_path / "analysis.ipynb")
    assert h.methods_critique["issues"][0]["kind"] == "fixable"
