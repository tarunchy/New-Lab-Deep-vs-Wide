from flask import Blueprint, render_template, request, redirect, url_for, send_file
from pathlib import Path
import json
import shutil
import matplotlib
matplotlib.use("Agg")

from .student import update_aggregate_with_student_scores
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

bp = Blueprint("main", __name__)
NOTES_ROOT = Path(__file__).parent.parent / "clinical_notes"
RUNS_ROOT = Path(__file__).parent.parent / "clinical_notes" / "runs"
NOTE_TOKEN_MAP = {
    "clinical_note1": 66416, "clinical_note2": 66921, "clinical_note3": 27655, "clinical_note4": 25510,
    "clinical_note5": 25494, "clinical_note6": 25192, "clinical_note7": 40191, "clinical_note8": 40382,
    "clinical_note9": 40144, "clinical_note10": 65920, "clinical_note11": 65863, "clinical_note12": 65851,
    "clinical_note13": 51875, "clinical_note14": 65868, "clinical_note15": 10221, "clinical_note16": 10068,
}

# ---------- Index and Student Answer Submission ----------
@bp.route("/")
def index():
    notes = sorted([p for p in NOTES_ROOT.iterdir() if p.is_dir() and p.name.startswith("clinical_note")],
                   key=lambda p: int(p.name.replace("clinical_note", "")))
    # For each note, load answer if exists
    answers = {}
    for note in notes:
        ans_file = note / "student_answer.txt"
        answers[note.name] = ans_file.read_text() if ans_file.exists() else ""
    return render_template("index.html", notes=notes, answers=answers)

@bp.route("/submit/<note_id>", methods=["POST"])
def submit(note_id):
    answer = request.form.get("student_answer", "").strip()
    note_dir = NOTES_ROOT / note_id
    (note_dir / "student_answer.txt").write_text(answer)
    # Copy to run1 (or specified run)
    run_name = request.form.get("run_name", "run1")
    run_dir = RUNS_ROOT / run_name / note_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "student_answer.txt").write_text(answer)
    return redirect(url_for('main.index'))

from flask import Response

@bp.route("/view_note/<note_id>")
def view_note(note_id):
    file_path = NOTES_ROOT / note_id / "clinical_note.txt"
    if not file_path.exists():
        return "File not found", 404
    return Response(file_path.read_text(), mimetype='text/plain')


@bp.route("/download/<note_id>/clinical_note.txt")
def download_note(note_id):
    file_path = NOTES_ROOT / note_id / "clinical_note.txt"
    if not file_path.exists():
        return "File not found", 404
    return send_file(file_path, as_attachment=True, download_name=f"{note_id}_clinical_note.txt")

@bp.route("/clean_all_student_answers")
def clean_all_student_answers():
    # Remove from both clinical_notes/clinical_note* and runs/run1/clinical_note*
    for note in NOTES_ROOT.iterdir():
        if note.is_dir() and note.name.startswith("clinical_note"):
            student_file = note / "student_answer.txt"
            if student_file.exists():
                student_file.unlink()
            # Also in run1
            run_note = RUNS_ROOT / "run1" / note.name / "student_answer.txt"
            if run_note.exists():
                run_note.unlink()
    # Remove score plot if exists
    score_plot = RUNS_ROOT / "run1" / "student_score_trend.png"
    if score_plot.exists():
        score_plot.unlink()

    # Remove student score summary JSON if exists
    score_json = RUNS_ROOT / "run1" / "student_score_summary.json"
    if score_json.exists():
        score_json.unlink()

    return redirect(url_for('main.index'))


# ---------- Clone Student Answer ----------
@bp.route("/clone_student_answer/<note_id>")
def clone_student_answer(note_id):
    # Copy answer from <note_id> to all remaining notes (those after note_id)
    source = NOTES_ROOT / note_id / "student_answer.txt"
    if not source.exists():
        return redirect(url_for('main.index'))
    notes = sorted([p for p in NOTES_ROOT.iterdir() if p.is_dir() and p.name.startswith("clinical_note")],
                   key=lambda p: int(p.name.replace("clinical_note", "")))
    start_copy = False
    for note in notes:
        if note.name == note_id:
            start_copy = True
            continue
        if start_copy:
            (note / "student_answer.txt").write_text(source.read_text())
            # Also copy to run1
            run_note = RUNS_ROOT / "run1" / note.name
            run_note.mkdir(parents=True, exist_ok=True)
            (run_note / "student_answer.txt").write_text(source.read_text())
    return redirect(url_for('main.index'))

# ---------- Run Visualizations ----------
@bp.route("/run_visuals/<run_name>")
def run_visuals(run_name):
    run_dir = RUNS_ROOT / run_name
    images = [
        "score_trend_by_token_size_cosine.png",
        "score_trend_by_token_size_rouge.png",
        "winner_trend_by_token_size_cosine.png",
        "winner_trend_by_token_size_rouge.png"
    ]
    available_images = [img for img in images if (run_dir / img).exists()]
    return render_template("run_visuals.html", run_name=run_name, images=available_images)

@bp.route("/run_img/<run_name>/<filename>")
def run_img(run_name, filename):
    file_path = (RUNS_ROOT / run_name / filename).resolve()
    print("Trying to serve file:", file_path)
    if not file_path.exists():
        return "File not found", 404
    return send_file(str(file_path))

# ---------- Score Student Answers and Visualize ----------
@bp.route("/student_score/<run_name>")
def student_score(run_name):
    results = score_student_answers(run_name)
    run_dir = RUNS_ROOT / run_name
    img_path = run_dir / "student_score_trend.png"
    student_score_img_exists = img_path.exists()
    return render_template(
        "student_score_visual.html",
        run_name=run_name,
        results=results,
        student_score_img_exists=student_score_img_exists
    )

def score_student_answers(run_name="run1"):
    base_dir = NOTES_ROOT
    run_dir = RUNS_ROOT / run_name
    cases = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("clinical_note")],
                   key=lambda p: int(p.name.split("clinical_note")[1]))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    cosine_scores, rouge_scores, tokens, cases_names = [], [], [], []

    results = []
    for case in cases:
        case_name = case.name
        base_note = base_dir / case_name
        student_path = case / "student_answer.txt"
        gold_path = base_note / "gold_standard_answer.txt"

        if student_path.exists() and gold_path.exists():
            student = student_path.read_text().strip()
            gold = gold_path.read_text().strip()
            emb1 = model.encode([student])[0]
            emb2 = model.encode([gold])[0]
            cosine = float(cosine_similarity([emb1], [emb2])[0][0])
            rouge = scorer.score(gold, student)['rougeL'].fmeasure
            cosine_scores.append(cosine)
            rouge_scores.append(rouge)
            student_ans = student
            gold_ans = gold
        else:
            cosine_scores.append(None)
            rouge_scores.append(None)
            student_ans = "Answer not submitted"
            gold_ans = gold_path.read_text().strip() if gold_path.exists() else "Not available"

        tokens.append(NOTE_TOKEN_MAP.get(case_name, 0))
        cases_names.append(case_name)
        results.append({
            "case": case_name,
            "cosine": cosine_scores[-1],
            "rougeL_f1": rouge_scores[-1],
            "student_answer": student_ans,
            "gold_answer": gold_ans
        })

    # Plotting
    plt.figure(figsize=(12, 5))
    valid = [i for i, s in enumerate(cosine_scores) if s is not None]
    if valid:
        plt.plot([tokens[i] for i in valid], [cosine_scores[i] for i in valid], marker='o', label="Cosine Similarity")
        plt.plot([tokens[i] for i in valid], [rouge_scores[i] for i in valid], marker='x', label="ROUGE-L F1")
        plt.xlabel("Note Token Count")
        plt.ylabel("Score")
        plt.legend()
        plt.title("Student Answer Similarity to Gold Standard")
        plt.tight_layout()
        plt.savefig(run_dir / "student_score_trend.png")
        plt.close()
    return results

# ---------- Aggregate (keep as before) ----------
@bp.route("/run_student_aggregate")
def run_student_aggregate():
    update_aggregate_with_student_scores()
    return redirect(url_for('main.summary'))

@bp.route("/summary")
def summary():
    with open(NOTES_ROOT / "runs" / "aggregate_summary.json") as f:
        summary_data = json.load(f)
    return render_template("summary.html", summary=summary_data)



import json
from flask import jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once at top
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def embedding_cosine_similarity(text1: str, text2: str) -> float:
    # This is a minimal function for scoring similarity
    if not text1 or not text2:
        return None
    emb1 = EMBED_MODEL.encode([text1])[0]
    emb2 = EMBED_MODEL.encode([text2])[0]
    return float(cosine_similarity([emb1], [emb2])[0][0])

@bp.route("/student_score_json")
def student_score_json():
    """
    Return JSON for all student submissions (and model answers), for plotting in JS.
    Example output:
    [
      {"case": "clinical_note1", "student": 0.82, "wide": 0.77, "rag": 0.7, "clear": 0.79},
      {"case": "clinical_note2", "student": null, ...},
      ...
    ]
    """
    summary = []
    run_dir = RUNS_ROOT / "run1"
    for note in sorted(
        [p for p in NOTES_ROOT.iterdir() if p.is_dir() and p.name.startswith("clinical_note")],
        key=lambda p: int(p.name.replace("clinical_note", ""))
    ):

        if not note.is_dir() or not note.name.startswith("clinical_note"):
            continue

        case_name = note.name
        student_file = note / "student_answer.txt"
        gold_file = note / "gold_standard_answer.txt"

        student_score = None
        student_answer = None

        # Get student answer + score if exists
        if student_file.exists() and gold_file.exists():
            student_answer = student_file.read_text().strip()
            gold_answer = gold_file.read_text().strip()
            student_score = embedding_cosine_similarity(student_answer, gold_answer)

        # Get model scores from run dir (use 0 if missing)
        model_scores = {}
        for strat in ("wide", "rag", "clear"):
            model_file = run_dir / case_name / f"answer_{strat}.txt"
            if model_file.exists() and gold_file.exists():
                model_ans = model_file.read_text().strip()
                model_scores[strat] = embedding_cosine_similarity(model_ans, gold_file.read_text().strip())
            else:
                model_scores[strat] = None

        summary.append({
            "case": case_name,
            "student": student_score,
            "wide": model_scores["wide"],
            "rag": model_scores["rag"],
            "clear": model_scores["clear"],
        })
    return jsonify(summary)

@bp.route("/student_score_plot")
def student_score_plot():
    # No data needed, the page fetches JSON from /student_score_json
    return render_template("student_score_plot.html")

from flask import jsonify, send_file
import io

@bp.route("/download_final_submission")
def download_final_submission():
    import json
    from datetime import datetime

    # Build the results structure
    submission = []
    for note in sorted(
        [p for p in NOTES_ROOT.iterdir() if p.is_dir() and p.name.startswith("clinical_note")],
        key=lambda p: int(p.name.replace("clinical_note", ""))
    ):
        case = note.name
        question_path = note / "question.txt"
        student_answer_path = note / "student_answer.txt"
        gold_path = note / "gold_standard_answer.txt"
        question = question_path.read_text().strip() if question_path.exists() else ""
        student_answer = student_answer_path.read_text().strip() if student_answer_path.exists() else ""
        gold_answer = gold_path.read_text().strip() if gold_path.exists() else ""

        # You may want to add scoring, if desired:
        cosine, rougeL_f1 = None, None
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            from rouge_score import rouge_scorer

            EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            ROUGE = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

            if student_answer and gold_answer:
                emb1 = EMBED_MODEL.encode([student_answer])[0]
                emb2 = EMBED_MODEL.encode([gold_answer])[0]
                cosine = float(cosine_similarity([emb1], [emb2])[0][0])
                rougeL_f1 = ROUGE.score(gold_answer, student_answer)['rougeL'].fmeasure
        except Exception as e:
            # If no torch/transformer etc, skip scoring and only export the answers
            pass

        submission.append({
            "case": case,
            "question": question,
            "student_answer": student_answer,
            "gold_answer": gold_answer,
            "cosine_similarity": cosine,
            "rougeL_f1": rougeL_f1,
        })

    # Create the download in-memory (no need to write to disk)
    export_filename = f"student_final_submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_bytes = json.dumps(submission, indent=2).encode("utf-8")
    return send_file(
        io.BytesIO(json_bytes),
        as_attachment=True,
        download_name=export_filename,
        mimetype="application/json"
    )

@bp.route("/final_submission_status")
def final_submission_status():
    notes = sorted(
        [p for p in NOTES_ROOT.iterdir() if p.is_dir() and p.name.startswith("clinical_note")],
        key=lambda p: int(p.name.replace("clinical_note", ""))
    )
    missing = []
    for note in notes:
        if not (note / "student_answer.txt").exists() or not (note / "student_answer.txt").read_text().strip():
            missing.append(note.name)
    return jsonify({"missing": missing})
