import json, math, csv, yaml
from pathlib import Path
from statistics import mean
from collections import Counter

CFG = yaml.safe_load(Path("experiment.yml").read_text())
RUNS_DIR = Path(CFG["runs_root"])
MARGIN = CFG["win_margin"]

BUCKETS = [
    ("VERY SMALL", 12_000),
    ("SMALL", 30_000),
    ("MEDIUM", 50_000),
    ("LARGE", math.inf)
]

def bucket_name(tok_count):
    for name, limit in BUCKETS:
        if tok_count < limit:
            return name
    return "LARGE"

def generate_model_summary():
    rows = []
    for csv_file in sorted(RUNS_DIR.rglob("experiment_summary.csv")):
        with csv_file.open() as f:
            rows.extend(list(csv.DictReader(f)))

    if not rows:
        return {}

    for row in rows:
        for key in ["Note Tokens", "Wide Accuracy", "RAG Accuracy", "CLEAR Accuracy"]:
            row[key] = float(row[key])
        row["Size Bucket"] = bucket_name(int(row["Note Tokens"]))

    def winner(row):
        base = row["Wide Accuracy"]
        best = "Wide"
        if row["RAG Accuracy"] - base > MARGIN:
            best, base = "RAG", row["RAG Accuracy"]
        if row["CLEAR Accuracy"] - base > MARGIN:
            best = "CLEAR"
        return best

    for r in rows:
        r["Winner"] = winner(r)

    summary = {}
    for name, _ in BUCKETS:
        group = [r for r in rows if r["Size Bucket"] == name]
        if not group: continue
        win_counts = Counter(r["Winner"] for r in group)
        avg_acc = {
            s: mean(r[f"{s} Accuracy"] for r in group)
            for s in ("Wide", "RAG", "CLEAR")
        }
        summary[name] = {
            "#cases": len(group),
            "wins": [win_counts["Wide"], win_counts["RAG"], win_counts["CLEAR"]],
            "avg_acc": [avg_acc["Wide"], avg_acc["RAG"], avg_acc["CLEAR"]],
        }

    out_path = RUNS_DIR / "aggregate_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"✅ Updated {out_path}")
    return summary
