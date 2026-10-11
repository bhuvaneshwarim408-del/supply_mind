"""Run every experiment and write results/experiments.json (used by the web app's Experiments page).

    python scripts/run_experiments.py            # full run (about 10-20 minutes)
    python scripts/run_experiments.py --quick    # fewer SKUs and episodes, about 2 minutes
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scm import backtest, data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/retail_store_inventory.csv")
    ap.add_argument("--out", default="results/experiments.json")
    ap.add_argument("--skus", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["real", "synthetic", "volatile"], help="run one dataset and merge into the output file")
    args = ap.parse_args()
    if args.quick:
        args.skus, args.episodes = 12, 1500
    out = {"datasets": {}}
    if args.only and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            out = json.load(fh)
    out["generated"] = datetime.now(timezone.utc).isoformat()
    panels = {
        "real": data.load_panel(args.csv, n_skus=args.skus, name="Retail CSV (real data)"),
        "synthetic": data.synthetic_panel(n_skus=args.skus, name="Structured synthetic (has patterns to learn)"),
        "volatile": data.synthetic_panel(n_skus=args.skus, name="Synthetic with volatile promotions", volatile_promo=True, seed=1),
    }
    for key, panel in panels.items():
        if args.only and key != args.only:
            continue
        out["datasets"][key] = backtest.run_dataset(panel, rl_episodes=args.episodes, log=lambda m: print(m, flush=True))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
