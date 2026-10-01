"""Executes dashboard.py against a minimal fake `streamlit` so that every page's Python logic runs.
This catches NameErrors / wrong calls / data errors. It does NOT test real Streamlit rendering."""
import sys, types
from pathlib import Path
from unittest.mock import MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Ctx(MagicMock):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def build(page, text="1200, 1203, 1203, 1199, 1195, 1195, 1190, 1188, 1188, 1185, 1180, 1181, 1176, 1170, 1170, 1166, 1160"):
    st = types.ModuleType("streamlit")
    cache = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    st.cache_data = cache; st.cache_resource = cache
    st.__getattr__ = None
    log = []
    def cols(spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [Ctx() for _ in range(n)]
    class Sidebar:
        def __getattr__(self, n):
            def f(*a, **k):
                if n == "file_uploader": return None
                if n == "radio": return page
                log.append((n, a[:1])); return None
            return f
    st.sidebar = Sidebar()
    class Col(Ctx):
        def selectbox(self, label, opts, **k): return st.selectbox(label, opts)
        def slider(self, label, lo, hi, v, **k): return v
        def checkbox(self, label, value=False, **k): return value
    st.columns = lambda spec, **k: [Col() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.selectbox = lambda label, opts, **k: "AAPL" if "AAPL" in list(opts) else list(opts)[0]
    st.slider = lambda label, lo, hi, v, **k: v
    st.checkbox = lambda label, value=False, **k: value
    st.button = lambda *a, **k: True
    st.text_area = lambda label, value="", **k: text
    st.expander = lambda *a, **k: Ctx()
    st.stop = lambda: (_ for _ in ()).throw(SystemExit("st.stop called"))
    for n in ("title caption header subheader write metric error warning info success dataframe pyplot image "
              "download_button set_page_config").split():
        setattr(st, n, lambda *a, _n=n, **k: log.append((_n, str(a[:1])[:90])))
    st._log = log
    return st


if __name__ == "__main__":
    import importlib
    ok = True
    for page, txt in [("Risk Analysis", None), ("Watchlist", None), ("Try your own data", None),
                      ("Try your own data", "1,2,3"), ("Model Performance", None), ("Explanations & Limitations", None)]:
        st = build(page) if txt is None else build(page, txt)
        sys.modules["streamlit"] = st
        sys.modules.pop("dashboard", None)
        try:
            d = importlib.import_module("dashboard"); d.main()
            errs = [l for l in st._log if l[0] == "error"]
            print("PASS" if (not errs or txt) else "FAIL", page, "| widgets:", len(st._log), "| errors shown:", [e[1] for e in errs])
            ok &= (not errs or bool(txt))
        except BaseException as e:
            ok = False; print("FAIL", page, repr(e)[:300])
    sys.exit(0 if ok else 1)
