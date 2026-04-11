<div align="center">

<img src="Schindler_2.png" alt="Schindler Logo" width="160"/>

# Schindler AI Maintenance Intelligence Platform

**Predictive maintenance for elevator fleets — powered by a Voting Ensemble of ML & Deep Learning models**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Internal%20Use-6B7A9F?style=flat-square)](./LICENSE)

[Features](#-features) · [Model Architecture](#-model-architecture) · [Quick Start](#-quick-start) · [Project Structure](#-project-structure) · [What to Upload](#-what-to-upload-to-github)

</div>

---

## Overview

The **Schindler AI Maintenance Intelligence Platform** is a full-stack web application that predicts elevator maintenance needs using a soft-voting ensemble of four ML/DL models trained on **23,617 real inspection records**. It provides role-based access for engineers and administrators, bulk fleet analysis, PDF/Excel reporting, and an AI-powered chatbot assistant (ARIA).

---

## Features

### Engineer Workflow

| Module | Description |
|---|---|
| **Single Unit Diagnostics** | Enter a technician checklist (Yes/No per feature) and get an instant risk score with a visual gauge |
| **Bulk Fleet Processor** | Upload an `.xlsx` fleet sheet — all units are scored, ranked, and named (Unit 1, Unit 2, ...) automatically |
| **Fleet Dashboard** | Auto-populated after any analysis. Shows risk pie, histogram, top-10 units, and a full scatter map |
| **ARIA Chatbot** | AI advisor (Llama 3.3-70B via Groq) with full fleet data context. Generates schedules, exec reports, and prioritized action plans |

### Admin Panel

| Feature | Description |
|---|---|
| **Command Center** | Platform-wide KPIs: total uploads, diagnostics, logins, units analyzed, and total critical units |
| **Cross-Engineer Analytics** | Units analyzed per engineer, platform-wide risk distribution, and avg risk trend over time |
| **Engineer Deep Dive** | Replay any engineer's upload session with full charts and export capability |
| **Activity Logs** | Per-engineer breakdown of all actions (logins, uploads, diagnostics, chatbot queries) |

---

## Model Architecture

> Trained on **23,617 records x 49 features** (46 safety checklist items + 3 engineered features)

### Ensemble

The production model is a **manual soft-voting ensemble** — it averages the probability outputs of all four models:

```
Random Forest  ──┐
Gradient Boost ──┤──▶  avg(prob)  ──▶  Risk Score  ──▶  Risk Tier
XGBoost        ──┤
Neural Network ──┘
```

### Neural Network Architecture

```
Input (49) → Dense(256, ReLU) → BatchNorm → Dropout(0.3)
           → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
           → Dense(64,  ReLU)
           → Dense(1, Sigmoid)
```

Trained with `EarlyStopping` on `val_auc` and `ReduceLROnPlateau`.

### Performance

| Metric | Score |
|---|---|
| AUC-ROC | **0.8373** |
| F1 Score | **0.7732** |
| Accuracy | 0.7405 |
| Decision Threshold | **0.3674** *(optimized on validation set — default 0.5 was not used)* |

### Risk Tiers

| Tier | Score Range | Recommended Action |
|---|---|---|
| Safe | 0 – 25% | Routine inspection on schedule |
| Low Risk | 25 – 50% | Monitor; inspect within 30 days |
| Medium Risk | 50 – 75% | Schedule maintenance within 2–4 weeks |
| Critical | 75 – 100% | Immediate action — evaluate taking unit offline |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / Backend | Streamlit |
| ML Models | scikit-learn (Random Forest, Gradient Boosting), XGBoost |
| Deep Learning | TensorFlow / Keras |
| AI Chatbot | Groq API · `langchain-groq` · Llama 3.3-70B |
| Database | SQLite (auth, sessions, activity logs) |
| Charts | Plotly Express & Graph Objects |
| PDF Export | fpdf2 |
| Excel Export | xlsxwriter, openpyxl |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/schindler-ai-platform.git
cd schindler-ai-platform
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit pandas numpy tensorflow scikit-learn xgboost \
            plotly fpdf2 xlsxwriter openpyxl langchain-groq
```

### 4. Place model artifacts

All model files must be in the **same directory as `app.py`**:

```
schindler-ai-platform/
├── app.py
├── scaler_v2.pkl
├── features_v2.pkl
├── imputer_v2.pkl
├── best_threshold.pkl
├── voting_ensemble.pkl
├── gradient_boosting_model.pkl
├── xgboost_model.pkl
├── maintenance_model_v2.keras
├── model_metadata.json
└── random_forest_model.pkl        ← see note below (190 MB)
```

> **Note on `random_forest_model.pkl` (190 MB):**
> This file exceeds GitHub's 100 MB limit. Download it from the [Releases](../../releases) page and place it in the root directory before running.

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 6. Default credentials

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `admin2026` |
| Engineer | Register via the app | — |

> **Change the admin password before any production deployment.**

### 7. Enable ARIA (AI Chatbot) — optional

1. Get a free API key at [console.groq.com](https://console.groq.com)
2. Open the sidebar → **API Configuration**
3. Paste your key (`gsk_...`)

---

## Project Structure

```
schindler-ai-platform/
│
├── app.py                            # Main Streamlit application (v5.0)
├── schindler_model_training.ipynb    # Full ML/DL training pipeline
│
├── model_metadata.json               # Model info and feature registry
├── best_threshold.pkl                # Optimized decision threshold (0.3674)
├── voting_ensemble.pkl               # Ensemble voting configuration
├── features_v2.pkl                   # Feature names used during training
├── scaler_v2.pkl                     # StandardScaler (for NN input)
├── imputer_v2.pkl                    # Missing value imputer
│
├── gradient_boosting_model.pkl       # Gradient Boosting classifier
├── xgboost_model.pkl                 # XGBoost classifier
├── maintenance_model_v2.keras        # Keras Neural Network
├── random_forest_model.pkl           # Random Forest (190 MB — see Releases)
│
├── Schindler.png                     # Primary logo
├── Schindler_2.png                   # Sidebar logo
│
├── schindler_platform.db             # Runtime SQLite database (git-ignored)
└── README.md
```

---

## What to Upload to GitHub

| File | Size | Upload? | Notes |
|---|---|---|---|
| `app.py` | 68 KB | Yes | Main application |
| `schindler_model_training.ipynb` | 559 KB | Yes | Full training pipeline |
| `model_metadata.json` | 1.4 KB | Yes | — |
| `best_threshold.pkl` | 21 B | Yes | — |
| `voting_ensemble.pkl` | 139 B | Yes | — |
| `features_v2.pkl` | 9 KB | Yes | — |
| `scaler_v2.pkl` | 10 KB | Yes | — |
| `imputer_v2.pkl` | 10 KB | Yes | — |
| `xgboost_model.pkl` | 524 KB | Yes | — |
| `gradient_boosting_model.pkl` | 970 KB | Yes | — |
| `maintenance_model_v2.keras` | 743 KB | Yes | — |
| `Schindler.png` / `Schindler_2.png` | ~4 MB total | Yes | UI assets |
| `random_forest_model.pkl` | **190 MB** | Release asset only | Exceeds GitHub 100 MB limit |
| `maintenance_dataset.xlsx` | 3.9 MB | No | Training data — keep private |
| `schindler_platform.db` | 52 KB | No | Runtime DB with user data |
| `venv/` | — | No | Never commit virtual environments |

### Recommended `.gitignore`

```gitignore
# Runtime database
schindler_platform.db

# Virtual environment
venv/
.env

# Python cache
__pycache__/
*.pyc
*.pyo

# Large model file (upload as GitHub Release asset)
random_forest_model.pkl

# Training data (keep private)
maintenance_dataset.xlsx

# OS artifacts
.DS_Store
Thumbs.db
```

### Handling the 190 MB Random Forest Model

**Option A — GitHub Releases (recommended)**

Upload `random_forest_model.pkl` manually as an asset when creating a GitHub Release. Then document the download link here for other contributors.

**Option B — Git LFS**

```bash
git lfs install
git lfs track "random_forest_model.pkl"
git add .gitattributes
git add random_forest_model.pkl
git commit -m "feat: add RF model via Git LFS"
git push
```

---

## Authentication & Security

- Admin account is seeded automatically on first run
- Engineer self-registration is capped at **15 engineers**
- Passwords are hashed with **SHA-256**
- All user activity is logged to SQLite: logins, uploads, diagnostics, chatbot queries
- The database file (`schindler_platform.db`) is git-ignored and never committed

---

## License

This project is developed for internal and educational purposes and is not affiliated with or endorsed by **Schindler Group**.
