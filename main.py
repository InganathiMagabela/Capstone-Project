"""Master script:  python main.py

validate project -> load config -> load + validate data -> preprocess -> build features
-> load/validate model -> evaluate on the test period -> (optional) retrain -> launch dashboard

Options:  --no-dashboard   run the pipeline checks only
          --retrain        retrain the model (notebook sections 6-10) into models/
          --tune           with --retrain, repeat the hyper-parameter search (slow)
          --data PATH      use a different holder-count CSV
          --port N         dashboard port
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import config as C                                   # noqa: E402
from pipelines.evaluation import evaluate_model                   # noqa: E402
from pipelines.feature_engineering import build_training_table   # noqa: E402
from pipelines.prediction import load_model                       # noqa: E402
from pipelines.preprocessing import find_default_data, load_and_validate  # noqa: E402
from pipelines.training import train_final_model                  # noqa: E402
from utils.helpers import environment_info, write_json            # noqa: E402
from utils.validation import validate_model_artifact              # noqa: E402


def step(msg):
    print(f"\n[{msg}]")


def validate_project():
    required = [ROOT / "dashboard.py", C.MODELS_DIR, C.ARTIFACTS_DIR]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"ERROR: project files missing: {missing}")
    if not (C.MODEL_FILE.exists() or C.PORTABLE_MODEL_FILE.exists()):
        print("Note: no saved model found - use --retrain to create one.")
    print("Project structure OK. Environment:", environment_info())


def launch_dashboard(port=8501):
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit("Streamlit is not installed. Run:  pip install -r requirements.txt")
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard.py"), "--server.port", str(port)]
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args(argv)

    step("1. Validate project"); validate_project()
    step("2. Load configuration")
    print(f"Target: {C.LABEL_MAP}  | features: {len(C.FEATURES)}  | split {C.TRAIN_FRAC}/{C.VAL_FRAC}/test  | seed {C.RANDOM_STATE}")

    step("3. Load and validate data")
    data_path = args.data or find_default_data()
    if data_path is None:
        raise SystemExit(f"ERROR: no data found. Place robin_dataset.csv in {C.DATA_RAW}.")
    clean, rep = load_and_validate(data_path)
    for w in rep.warnings:
        print("  warning:", w)
    if not rep.ok:
        raise SystemExit("ERROR:\n  " + "\n  ".join(rep.errors))
    print(f"Data OK: {rep.info['rows']:,} raw rows | {rep.info['stocks']} stocks | {rep.info['start']} -> {rep.info['end']}")

    step("4-5. Preprocess and build features")
    table = build_training_table(clean)
    print(f"Modelling table: {len(table):,} rows x {len(C.FEATURES)} features; decrease rate {table['risk'].mean():.2%}")

    if args.retrain:
        step("Retraining (chronological split, validation threshold)")
        art = train_final_model(clean, tune=args.tune)
        print(f"Saved new model to {C.MODELS_DIR} (threshold {art['threshold']:.4f}).")

    step("6. Load and validate model")
    art, msg = load_model()
    print(msg)
    mrep = validate_model_artifact(art)
    if not mrep.ok:
        raise SystemExit("ERROR: model incompatible:\n  " + "\n  ".join(mrep.errors))
    print(f"Model '{art['model_name']}' compatible: {len(art['features'])} features, threshold {art['threshold']:.4f}")

    step("7. Evaluate on the chronological test period")
    res = evaluate_model(art, clean)
    write_json(C.LIVE_METRICS_FILE, res)
    print({k: round(v, 4) for k, v in res["metrics"].items()}, f"| test rows {res['test_rows']:,}")
    print(f"Metrics saved to {C.LIVE_METRICS_FILE.relative_to(ROOT)}")

    if args.no_dashboard:
        print("\nPipeline complete (dashboard not launched).")
        return 0
    step("8. Launch dashboard"); launch_dashboard(args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
