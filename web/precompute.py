"""Pre-train forecasts so the dashboard and pages load instantly.

Run from the project root:
    python -m web.precompute --n 5            # first 5 SKUs
    python -m web.precompute S001::P0001 S001::P0002 --explain
Each SKU takes about a minute the first time.
"""
import argparse

from web.services import data_source, real_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skus", nargs="*", help="SKU ids (default: the first --n SKUs)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--explain", action="store_true", help="also fetch the LLM explanation")
    args = ap.parse_args()
    if data_source.IS_DEMO:
        raise SystemExit("Real modules or the CSV were not found, so there is nothing to pre-train. Run from the project root.")
    targets = args.skus or data_source.SKUS[: args.n]
    for i, sku in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {sku} ...", flush=True)
        plan = real_backend.get_plan(sku, real_backend.DEFAULT_HORIZON, real_backend.DEFAULT_CONSTRAINTS)
        if args.explain:
            real_backend.explain(plan)
    print("Done. Start the app with: python run_web.py")


if __name__ == "__main__":
    main()
