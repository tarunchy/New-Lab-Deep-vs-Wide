import json
import math
from pathlib import Path
from statistics import mean
import numpy as np
import tiktoken
from rouge_score import rouge_scorer

ENC = tiktoken.encoding_for_model("gpt-4")
ROOT = Path("clinical_notes")
SUMMARY_FILE = ROOT / "runs" / "aggregate_summary.json"
ROUGE = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def embed(texts):
    # Dummy embed; replace with your real embeddings if available
    return [np.random.rand(512) for _ in texts]

def rouge_l_f1(pred, ref):
    return ROUGE.score(ref, pred)['rougeL'].fmeasure

def bucket_name(tok_count):
    if tok_count < 12_000:
        return "VERY SMALL"
    elif tok_count < 30_000:
        return "SMALL"
    elif tok_count < 50_000:
        return "MEDIUM"
    else:
        return "LARGE"

def update_aggregate_with_student_scores():
    notes = [p for p in ROOT.iterdir() if p.name.startswith("clinical_note") and p.is_dir()]
    buckets_cos = {}
    buckets_rouge = {}

    for note in notes:
        if not (note / "student_answer.txt").exists():
            continue
        note_txt = (note / "clinical_note.txt").read_text()
        gold_txt = (note / "gold_standard_answer.txt").read_text()
        student_txt = (note / "student_answer.txt").read_text()

        embeddings = embed([gold_txt, student_txt])
        cos_score = cos(embeddings[0], embeddings[1])
        rouge_score = rouge_l_f1(student_txt, gold_txt)

        bname = bucket_name(len(ENC.encode(note_txt)))
        buckets_cos.setdefault(bname, []).append(cos_score)
        buckets_rouge.setdefault(bname, []).append(rouge_score)

    if not SUMMARY_FILE.exists():
        print("Summary file not found!")
        return

    data = json.loads(SUMMARY_FILE.read_text())
    for bucket in data:
        # cosine
        if bucket in buckets_cos and buckets_cos[bucket]:
            data[bucket]["student_avg_cosine"] = round(mean(buckets_cos[bucket]), 4)
            data[bucket]["student_cases"] = len(buckets_cos[bucket])
        # rouge
        if bucket in buckets_rouge and buckets_rouge[bucket]:
            data[bucket]["student_avg_rouge"] = round(mean(buckets_rouge[bucket]), 4)

    SUMMARY_FILE.write_text(json.dumps(data, indent=2))
    print(f"Updated {SUMMARY_FILE} with student cosine & ROUGE scores.")

