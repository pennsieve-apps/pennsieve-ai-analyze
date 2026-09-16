import nbformat
import pytest

from analyze.agents.agentic_analyst import validate_notebook


@pytest.mark.parametrize("state", ["empty", "unexecuted", "error", "executed"])
def test_notebook_requires_executed_cells(tmp_path, state):
    cells = [] if state == "empty" else [nbformat.v4.new_code_cell("print(1)")]
    if state in {"error", "executed"}:
        cells[0].execution_count = 1
    if state == "error":
        cells[0].outputs = [nbformat.v4.new_output("error", ename="ValueError", evalue="bad data", traceback=[])]
    path = tmp_path / "analysis.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)
    if state == "executed":
        validate_notebook(path)
    else:
        with pytest.raises(RuntimeError):
            validate_notebook(path)
