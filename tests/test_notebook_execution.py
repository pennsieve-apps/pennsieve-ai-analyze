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


def test_a_markdown_cell_left_with_code_fields_is_tidied_not_failed(tmp_path):
    import json
    code = nbformat.v4.new_code_cell("print(1)")
    code.execution_count = 1
    notebook = nbformat.v4.new_notebook(cells=[code, nbformat.v4.new_markdown_cell("notes")])
    path = tmp_path / "analysis.ipynb"
    nbformat.write(notebook, path)
    raw = json.loads(path.read_text())
    raw["cells"][1].update(execution_count=3, outputs=[])   # a code cell retyped as markdown
    path.write_text(json.dumps(raw))
    validate_notebook(path)
    tidied = json.loads(path.read_text())["cells"][1]
    assert "execution_count" not in tidied and "outputs" not in tidied
    assert "".join(tidied["source"]) == "notes"


def test_tidying_does_not_hide_a_real_failure(tmp_path):
    import json
    code = nbformat.v4.new_code_cell("print(1)")   # never executed
    notebook = nbformat.v4.new_notebook(cells=[code, nbformat.v4.new_markdown_cell("notes")])
    path = tmp_path / "analysis.ipynb"
    nbformat.write(notebook, path)
    raw = json.loads(path.read_text())
    raw["cells"][1].update(execution_count=3, outputs=[])
    path.write_text(json.dumps(raw))
    with pytest.raises(RuntimeError, match="unexecuted"):
        validate_notebook(path)
