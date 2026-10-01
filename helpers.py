"""Small shared helpers."""
import json
import platform
from pathlib import Path


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def environment_info():
    import sklearn, pandas, numpy
    return {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "pandas": pandas.__version__, "numpy": numpy.__version__}


def pct(x, digits=1):
    return "n/a" if x is None or x != x else f"{x:.{digits}%}"
