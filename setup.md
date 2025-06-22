# ✅ Lab Setup Instructions

Welcome to the **Lab: Deep vs Wide – Clinical NLP**. Follow the steps below to get your local environment up and running quickly and consistently across **macOS, Windows, and Linux**.

---

## 🧠 Environment Overview

This lab was **developed and tested** with:

| Component        | Version / Details           |
|------------------|-----------------------------|
| macOS            | **15.3.1 (Sonoma)**         |
| Chip             | **Apple M1 (arm64)**        |
| Python           | **3.13.2**                  |
| Flask App Port   | `http://127.0.0.1:5001`     |
| Environment Tool | `venv` (built-in virtual env) |
| Requirements     | `requirements.txt` included |

---

## 🔧 Prerequisites

Make sure the following are installed:

- Python 3.10+ (tested on 3.13)
- Git
- `make` (optional for automation)
- OS: macOS (Intel/M1), Windows, or Linux

Check Python version:

```
python3 --version
```



## 🚀 Setup Steps

### 1. Clone the Repository

```
git clone https://github.com/YOUR_ORG/Lab-Deep-vs-Wide.git
cd Lab-Deep-vs-Wide
```

### 2. Create a Virtual Environment

```
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install Required Dependencies

```
pip install -r requirements.txt
```

### 4. Run the Flask App

```
python run.py
```

Access the app at: `http://127.0.0.1:5001`

---

## 🧪 Testing

1. Open the app in your browser: `http://localhost:5001`
2. Submit or view clinical notes
3. Use **Run Visuals**, **Download**, etc.
4. Ensure no notes are marked missing before downloading the final submission.

---

## 🆘 Troubleshooting

### ❓ Python not found?

Install with:

**macOS:**

```
brew install python
```

**Ubuntu:**

```
sudo apt update && sudo apt install python3 python3-venv
```

### ❓ Apple M1 and Torch Issues?

Force install CPU-only torch version:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 📦 `.gitignore` Example

Add this to avoid committing unnecessary files:

```
__pycache__/
.venv/
*.pyc
*.pkl
.env
```

---

## ✅ You're Ready!

Need help? Open an issue in the repo or contact your instructor.
