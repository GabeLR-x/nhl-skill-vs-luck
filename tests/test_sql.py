import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_window_function_queries_match_pandas():
    spec = importlib.util.spec_from_file_location("check_analysis", ROOT / "sql" / "check_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0
