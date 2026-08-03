import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "output/jupyter-notebook/voo-ema20-200-webull-tutorial.ipynb"


def test_voo_tutorial_notebook_contract() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    assert notebook["nbformat"] == 4
    assert cells and all(cell.get("id") for cell in cells)
    text = "\n".join("".join(cell.get("source", [])) for cell in cells)
    for required in (
        "VOO",
        "EMA",
        "WEBULL_APP_SECRET",
        "run_backtest",
        "Performance dashboard",
        "monthly return heatmap",
        "forward-testing selection",
        "run_forwardtesting",
        "Exercise",
    ):
        assert required in text
    assert "your_app_secret" in text
    assert "ghp_" not in text
    assert "github_pat_" not in text
