"""
Schindler AI Maintenance Intelligence Platform — v5.0
======================================================
• SQLite-backed auth: Admin login + Engineer register/login
• Admin: sees all engineers, their uploads, insights per engineer
• Engineer: bulk processor with unit numbering + charts, single diagnostics
• Smart ARIA chatbot with full data context
• Dashboard auto-populated from bulk or single
• Sidebar toggle: hide/show with ☰ and ✕ buttons
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, json, io, re, hashlib, warnings
from datetime import datetime

import tensorflow as tf
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
import sqlite3

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Schindler AI Platform",
    layout="wide",
    page_icon="🏢",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "schindler_platform.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT UNIQUE NOT NULL,
            name      TEXT NOT NULL,
            password  TEXT NOT NULL,
            role      TEXT NOT NULL DEFAULT 'engineer',
            created   TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            action      TEXT NOT NULL,
            detail      TEXT,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS upload_sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            filename      TEXT NOT NULL,
            total_units   INTEGER NOT NULL,
            critical      INTEGER DEFAULT 0,
            medium        INTEGER DEFAULT 0,
            low_risk      INTEGER DEFAULT 0,
            safe          INTEGER DEFAULT 0,
            avg_risk      REAL DEFAULT 0,
            result_json   TEXT,
            timestamp     TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS single_diag_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            risk_level  TEXT NOT NULL,
            probability REAL NOT NULL,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    admin_pw = hash_pw("admin2026")
    c.execute("""
        INSERT OR IGNORE INTO users (username, name, password, role, created)
        VALUES (?, ?, ?, 'admin', ?)
    """, ("admin", "Fleet Administrator", admin_pw, datetime.now().isoformat()))
    c.execute("""
        UPDATE users SET password=? WHERE username='admin'
    """, (admin_pw,))
    conn.commit()
    conn.close()

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def db_get_user(username: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None

def db_register_engineer(username, name, password):
    if db_get_user(username):
        return False, "Username already exists."
    conn = get_db()
    existing = conn.execute("SELECT COUNT(*) FROM users WHERE role='engineer'").fetchone()[0]
    conn.close()
    if existing >= 15:
        return False, "Maximum 15 engineers allowed. Contact admin."
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO users (username, name, password, role, created) VALUES (?,?,?,?,?)",
            (username, name, hash_pw(password), "engineer", datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        return True, "Registered successfully."
    except Exception as e:
        return False, str(e)

def db_log_activity(user_id: int, action: str, detail: str = ""):
    conn = get_db()
    conn.execute(
        "INSERT INTO activity_log (user_id, action, detail, timestamp) VALUES (?,?,?,?)",
        (user_id, action, detail, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def db_save_upload(user_id, filename, df: pd.DataFrame):
    dist = df["Risk Level"].value_counts().to_dict()
    save_cols = ["Unit Name","Risk Level","Maintenance Probability"]
    save_cols = [c for c in save_cols if c in df.columns]
    result_json = df[save_cols].to_json(orient="records")
    conn = get_db()
    conn.execute("""
        INSERT INTO upload_sessions
        (user_id, filename, total_units, critical, medium, low_risk, safe, avg_risk, result_json, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id, filename, len(df),
        dist.get("Critical", 0), dist.get("Medium Risk", 0),
        dist.get("Low Risk", 0), dist.get("Safe", 0),
        float(df["Maintenance Probability"].mean()),
        result_json, datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

def db_save_single_diag(user_id, risk_level, probability):
    conn = get_db()
    conn.execute(
        "INSERT INTO single_diag_log (user_id, risk_level, probability, timestamp) VALUES (?,?,?,?)",
        (user_id, risk_level, probability, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def db_get_all_engineers():
    conn = get_db()
    rows = conn.execute("SELECT * FROM users WHERE role='engineer' ORDER BY created DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_engineer_uploads(user_id: int):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM upload_sessions WHERE user_id=? ORDER BY timestamp DESC", (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_engineer_activity(user_id: int):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM activity_log WHERE user_id=? ORDER BY timestamp DESC LIMIT 40", (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_engineer_single_diags(user_id: int):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM single_diag_log WHERE user_id=? ORDER BY timestamp DESC LIMIT 50", (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_platform_stats():
    conn = get_db()
    stats = {
        "total_uploads": conn.execute("SELECT COUNT(*) FROM upload_sessions").fetchone()[0],
        "total_diags":   conn.execute("SELECT COUNT(*) FROM single_diag_log").fetchone()[0],
        "total_logins":  conn.execute("SELECT COUNT(*) FROM activity_log WHERE action='LOGIN'").fetchone()[0],
    }
    conn.close()
    return stats

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════════════
   SCHINDLER AI PLATFORM — DESIGN SYSTEM v6.0
   Industrial intelligence suite · dark navy / charcoal foundation
   ═══════════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:        #0a0d12;
    --bg-side:   #0e1219;
    --panel:     #121722;
    --panel-2:   #161c29;
    --line:      #232b3a;
    --line-soft: #1a2231;
    --ink:       #e8ecf3;
    --ink-2:     #a3aec0;
    --muted:     #64718a;
    --accent:    #4d8dff;
    --accent-dp: #3a6fe0;
    --safe:      #2fbf71;
    --low:       #d9a521;
    --med:       #ec7f37;
    --crit:      #e5484d;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
#MainMenu {visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden;}

.stApp {
    background:
        radial-gradient(1100px 520px at 75% -12%, rgba(77,141,255,0.075), transparent 62%),
        radial-gradient(900px 480px at -10% 108%, rgba(47,191,113,0.045), transparent 58%),
        var(--bg);
    color: var(--ink);
}
.stApp h1, .stApp h2, .stApp h3 { color: var(--ink); letter-spacing: -0.01em; }
.stApp a { color: var(--accent); }
code {
    background: #1a2130 !important; color: #9fc1ff !important;
    border-radius: 4px; padding: 1px 6px !important; font-size: 0.85em;
}
hr { border-color: var(--line-soft) !important; margin: 1.4rem 0 !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #242e40; border-radius: 8px; border: 2px solid var(--bg); }
::-webkit-scrollbar-thumb:hover { background: #33405c; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-side) !important;
    border-right: 1px solid var(--line-soft);
}
[data-testid="stSidebar"] hr { margin: .9rem 0 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 7px 10px; margin: 1px 0; border-radius: 8px;
    transition: background .15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(77,141,255,0.07); }
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    background-color: transparent !important; border-color: #3a465e !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover > div:first-child {
    border-color: var(--accent) !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] > div:first-child {
    border-color: var(--accent) !important;
}

/* ── Collapse control ── */
[data-testid="collapsedControl"] {
    display: flex !important; visibility: visible !important; opacity: 1 !important;
    background: var(--panel) !important; border: 1px solid var(--line) !important;
    border-left: none !important; border-radius: 0 8px 8px 0 !important;
    width: 28px !important; min-width: 28px !important; height: 56px !important;
    align-items: center !important; justify-content: center !important;
    cursor: pointer !important; top: 50% !important; transform: translateY(-50%) !important;
    box-shadow: 3px 0 12px rgba(0,0,0,0.45) !important;
    transition: background .2s, box-shadow .2s !important;
    z-index: 9999 !important; position: fixed !important;
}
[data-testid="collapsedControl"]:hover { background: #1c2434 !important; }
[data-testid="collapsedControl"] svg { fill: #7d90b8 !important; width: 16px !important; height: 16px !important; }

.sidebar-open-btn {
    position: fixed; top: 50%; left: 0; transform: translateY(-50%); z-index: 9999;
    background: var(--panel); border: 1px solid var(--line); border-left: none;
    border-radius: 0 8px 8px 0; width: 28px; height: 56px;
    display: flex; align-items: center; justify-content: center; cursor: pointer;
    box-shadow: 3px 0 12px rgba(0,0,0,0.45); transition: background .2s;
    color: #7d90b8; font-size: 16px; text-decoration: none;
}
.sidebar-open-btn:hover { background: #1c2434; }
.sidebar-close-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }

/* ── Page section headers (rendered via .section-header) ── */
.section-header {
    display: flex; align-items: center; gap: 14px;
    font-size: 1.42rem; font-weight: 700; color: #f2f5fa;
    letter-spacing: -0.015em; line-height: 1.2;
    padding: 4px 0 14px; margin: 4px 0 22px;
    border-bottom: 1px solid var(--line-soft);
}
.section-header::before {
    content: ""; display: inline-block; width: 4px; height: 26px; border-radius: 3px;
    background: linear-gradient(180deg, var(--accent), #2c4f96);
    box-shadow: 0 0 14px rgba(77,141,255,0.45);
}
.section-header:not(:has(*)):not(:empty)::after { content: none; }

/* ── Cards ── */
.info-card {
    background: linear-gradient(180deg, var(--panel-2), var(--panel));
    border: 1px solid var(--line); border-radius: 10px;
    padding: 20px 24px; margin-bottom: 16px;
    box-shadow: 0 1px 0 rgba(255,255,255,0.02) inset, 0 8px 22px rgba(0,0,0,0.28);
}
.info-card .label-sm { margin-bottom: 8px; }
.label-sm {
    font-size: 0.7rem; font-weight: 600; color: var(--muted);
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 4px;
}

/* ── Status banner (diagnostics result) ── */
.status-banner {
    padding: 16px 24px; border-radius: 10px; text-align: center;
    font-size: 1rem; font-weight: 600; letter-spacing: 0.06em; margin: 18px 0;
    display: flex; align-items: center; justify-content: center; gap: 14px;
    font-family: 'IBM Plex Mono', monospace;
    box-shadow: 0 10px 26px rgba(0,0,0,0.3);
}
.status-banner::before {
    content: ""; width: 10px; height: 10px; border-radius: 50%;
    background: currentColor; box-shadow: 0 0 12px currentColor; flex: none;
}

/* ── Metrics ── */
div[data-testid="metric-container"], [data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--panel-2), var(--panel));
    border: 1px solid var(--line); border-top: 2px solid #2a3548;
    border-radius: 10px; padding: 16px 18px 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    transition: border-color .2s ease, transform .2s ease;
}
div[data-testid="metric-container"]:hover, [data-testid="stMetric"]:hover {
    border-top-color: var(--accent); transform: translateY(-1px);
}
div[data-testid="metric-container"] label, [data-testid="stMetricLabel"] p {
    color: var(--muted) !important; font-size: 0.68rem !important; font-weight: 600 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="stMetricValue"] {
    color: #eef2f9 !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.72rem !important; font-weight: 600 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }
[data-testid="stMetric"] [data-testid="stMetricLabel"] { margin-bottom: 2px; }

/* ── Buttons ── */
.stButton>button, .stDownloadButton>button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 0.02em !important;
    border-radius: 8px !important; border: 1px solid transparent !important;
    transition: all .18s ease !important;
}
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] {
    background: linear-gradient(180deg, #5b97ff, var(--accent-dp)) !important;
    color: #ffffff !important; border: 1px solid #5b97ff !important;
    box-shadow: 0 4px 16px rgba(58,111,224,0.35) !important;
}
.stButton>button[kind="primary"]:hover, .stDownloadButton>button[kind="primary"]:hover {
    filter: brightness(1.09); transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(58,111,224,0.45) !important;
}
.stButton>button[kind="secondary"], .stDownloadButton>button[kind="secondary"] {
    background: rgba(255,255,255,0.02) !important; color: #c6d0e2 !important;
    border: 1px solid #2a3548 !important;
}
.stButton>button[kind="secondary"]:hover {
    border-color: var(--accent) !important; color: #ffffff !important;
}
.stDownloadButton>button { background: rgba(255,255,255,0.02) !important; color: #c6d0e2 !important; border: 1px solid #2a3548 !important; }
.stDownloadButton>button:hover { border-color: var(--accent) !important; color: #fff !important; }

/* ── Text inputs & selects ── */
[data-testid="stTextInput"] input, [data-baseweb="textarea"] {
    background: #0f141d !important; border: 1px solid var(--line) !important;
    border-radius: 8px !important; color: var(--ink) !important;
    transition: border-color .15s ease, box-shadow .15s ease !important;
}
[data-testid="stTextInput"] input:focus, [data-baseweb="textarea"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(77,141,255,0.18) !important;
}
[data-testid="stTextInput"] input::placeholder, [data-baseweb="textarea"] textarea::placeholder {
    color: #4d5a72 !important;
}
[data-baseweb="select"] > div {
    background: #0f141d !important; border: 1px solid var(--line) !important;
    border-radius: 8px !important; color: var(--ink) !important;
}
[data-baseweb="select"] > div:hover { border-color: #33405c !important; }
[data-baseweb="popover"] [role="listbox"] {
    background: var(--panel-2) !important; border: 1px solid var(--line) !important;
}
[data-baseweb="popover"] [role="option"]:hover { background: rgba(77,141,255,0.12) !important; }
[data-testid="stSelectbox"] label, [data-testid="stTextInput"] label { color: var(--ink-2) !important; font-weight: 500 !important; }

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: #0f141d !important; border: 1px dashed #33405c !important;
    border-radius: 10px !important; padding: 26px !important;
    transition: border-color .18s ease, background .18s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important; background: #101624 !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: transparent !important; color: var(--accent) !important;
    border: 1px solid #2a3548 !important; border-radius: 6px !important;
}
[data-testid="stFileUploader"] span { color: var(--ink-2) !important; }
[data-testid="stFileUploader"] small { color: var(--muted) !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--line) !important; gap: 6px !important;
}
[data-baseweb="tab"] {
    color: #8a97ad !important; font-weight: 500 !important;
    border-radius: 8px 8px 0 0 !important; padding: 8px 14px !important;
}
[data-baseweb="tab"]:hover { color: #c6d0e2 !important; background: rgba(255,255,255,0.03) !important; }
[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important; font-weight: 600 !important;
}
[data-baseweb="tab-highlight"] { background: var(--accent) !important; height: 2px !important; border-radius: 2px !important; }
[data-baseweb="tab-border"] { background-color: var(--line) !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid var(--line) !important; border-radius: 10px !important;
    background: var(--panel) !important; overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-weight: 500 !important; color: #dbe2ee !important; padding: 6px 4px !important;
}
[data-testid="stExpender"] details { border: none !important; }
[data-testid="stExpander"] details[open] summary { border-bottom: 1px solid var(--line-soft) !important; }

/* ── Data frames ── */
[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] {
    border: 1px solid var(--line) !important; border-radius: 10px !important; overflow: hidden;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important; border: 2px solid #0f141d !important;
    box-shadow: 0 0 0 3px rgba(77,141,255,0.25) !important;
}
[data-testid="stSlider"] > div[data-testid="stSliderThumb"] > div { background: var(--accent) !important; }
[data-testid="stSlider"] [data-testid="stSliderTrack"] > div { background: #242e40 !important; }

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: var(--panel) !important; border: 1px solid var(--line) !important;
    border-radius: 10px !important; margin-bottom: 8px !important;
}
[data-testid="stChatInput"] > div {
    background: var(--panel-2) !important; border: 1px solid var(--line) !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--ink) !important; background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #4d5a72 !important; }
[data-testid="stChatInput"] button { color: var(--accent) !important; }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(77,141,255,0.15) !important; }

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.03); border: 1px solid var(--line);
    border-left-width: 3px; border-radius: 8px; color: var(--ink-2);
}

/* ── Spinner ── */
[data-testid="stSpinner"], .stSpinner { color: var(--ink-2) !important; }
[data-testid="stStatusWidget"] div { color: var(--ink-2); }

/* ── Auth screen ── */
.auth-wrap { text-align: center; padding: 44px 0 26px; }
.auth-monogram {
    width: 58px; height: 58px; margin: 0 auto 14px; border-radius: 14px;
    background: linear-gradient(145deg, #1b2436, #10141d);
    border: 1px solid #2a3548; display: flex; align-items: center; justify-content: center;
    box-shadow: 0 12px 34px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
}
.auth-monogram span {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.7rem; font-weight: 600;
    background: linear-gradient(135deg, #7aa5ff, #3a6fe0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.auth-wordmark {
    font-size: 2.15rem; font-weight: 700; color: #f2f5fa;
    letter-spacing: 0.22em; text-indent: 0.22em;
}
.auth-tagline {
    display: inline-block; margin-top: 8px; padding: 4px 14px; border-radius: 999px;
    border: 1px solid var(--line); background: rgba(255,255,255,0.02);
    font-size: 0.68rem; color: #8a97ad; letter-spacing: 0.24em; text-transform: uppercase;
}
.auth-hint {
    font-size: 0.72rem; color: var(--muted); text-align: center; margin-top: 14px;
}
.auth-hint kbd {
    font-family: 'IBM Plex Mono', monospace; background: #1a2130; border: 1px solid #2a3548;
    border-bottom-width: 2px; border-radius: 4px; padding: 1px 6px; color: #9fc1ff; font-size: 0.68rem;
}

/* ── Sidebar identity card ── */
.side-user-card {
    margin: 14px 0 6px; padding: 13px 15px; border-radius: 10px;
    background: linear-gradient(180deg, var(--panel-2), var(--panel));
    border: 1px solid var(--line);
}
.side-user-card .u-name { font-size: 0.9rem; color: #eef2f9; font-weight: 600; }
.side-user-card .u-role {
    display: inline-block; margin-top: 6px; padding: 2px 9px; border-radius: 4px;
    font-size: 0.64rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
}
.side-status { font-size: 0.78rem; display: flex; align-items: center; gap: 7px; margin: 5px 0; }
.side-status .dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }
.side-version {
    font-size: 0.66rem; color: #39435c; margin-top: 10px; text-align: center; letter-spacing: 0.08em;
}

/* ── Context chips (chatbot) ── */
.ctx-chip {
    font-size: 0.75rem; margin-bottom: 14px; padding: 8px 16px;
    border-radius: 8px; border: 1px solid var(--line);
}
.ctx-chip.ok { color: #7bd8a8; background: rgba(47,191,113,0.07); border-color: rgba(47,191,113,0.25); }
.ctx-chip.off { color: var(--ink-2); background: var(--panel); }

/* ── Success banner (bulk complete) ── */
.bulk-done {
    background: rgba(47,191,113,0.08); border: 1px solid rgba(47,191,113,0.3);
    border-radius: 10px; padding: 12px 20px; font-size: 0.86rem;
    color: #7bd8a8; margin: 12px 0;
}

/* ── Admin panel ── */
.page-kicker {
    font-size: 0.68rem; font-weight: 600; color: var(--muted);
    letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 4px;
}
.page-title { font-size: 1.35rem; font-weight: 700; color: #f2f5fa; letter-spacing: -0.01em; }
.page-sub { font-size: 0.78rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 3px; }
.admin-profile {
    background: linear-gradient(180deg, var(--panel-2), var(--panel));
    border: 1px solid var(--line); border-radius: 12px;
    padding: 20px 28px; margin: 16px 0;
    display: flex; gap: 48px; align-items: center; flex-wrap: wrap;
    box-shadow: 0 8px 24px rgba(0,0,0,0.28);
}
.admin-stat { border-left: 1px solid var(--line); padding-left: 40px; display: flex; gap: 36px; flex-wrap: wrap; }
.session-strip {
    background: rgba(19,22,31,0.85); border: 1px solid var(--line); border-radius: 10px;
    padding: 14px 24px; margin-bottom: 18px; display: flex; gap: 40px;
    align-items: center; flex-wrap: wrap;
}
.kv .k { font-size: 0.66rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RISK CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RISK_CONFIG = {
    "Safe":        {"color":"#2a7f4f", "bg":"rgba(42,127,79,0.08)"},
    "Low Risk":    {"color":"#b8860b", "bg":"rgba(184,134,11,0.08)"},
    "Medium Risk": {"color":"#c06000", "bg":"rgba(192,96,0,0.08)"},
    "Critical":    {"color":"#b02020", "bg":"rgba(176,32,32,0.08)"},
}
RISK_RECS = {
    "Safe":        "No immediate action required. Schedule next routine inspection per standard calendar.",
    "Low Risk":    "Monitor closely. Flag for inspection at next available service window within 30 days.",
    "Medium Risk": "Schedule maintenance within 2–4 weeks. Prioritize safety circuit and door mechanism checks.",
    "Critical":    "Immediate maintenance required. Evaluate taking unit out of service pending inspection.",
}

def prob_to_risk(prob: float) -> str:
    if prob < 0.25: return "Safe"
    if prob < 0.50: return "Low Risk"
    if prob < 0.75: return "Medium Risk"
    return "Critical"

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class ElevatorRiskModel:
    def __init__(self):
        self.nn_model=None; self.rf_model=None; self.gb_model=None; self.xgb_model=None
        self.scaler=None; self.features=None; self.metadata={}
        self.threshold=0.5
        self._load()

    def _load(self):
        try:
            if os.path.exists("scaler_v2.pkl"):
                with open("scaler_v2.pkl","rb") as f: self.scaler = pickle.load(f)
            if os.path.exists("features_v2.pkl"):
                with open("features_v2.pkl","rb") as f: self.features = pickle.load(f)
            if os.path.exists("best_threshold.pkl"):
                with open("best_threshold.pkl","rb") as f: self.threshold = pickle.load(f)
            if os.path.exists("random_forest_model.pkl"):
                with open("random_forest_model.pkl","rb") as f: self.rf_model = pickle.load(f)
            if os.path.exists("gradient_boosting_model.pkl"):
                with open("gradient_boosting_model.pkl","rb") as f: self.gb_model = pickle.load(f)
            if os.path.exists("xgboost_model.pkl"):
                with open("xgboost_model.pkl","rb") as f: self.xgb_model = pickle.load(f)
            if os.path.exists("maintenance_model_v2.keras"):
                self.nn_model = tf.keras.models.load_model("maintenance_model_v2.keras")
            if os.path.exists("model_metadata.json"):
                with open("model_metadata.json") as f: self.metadata = json.load(f)
        except Exception as e:
            st.error(f"Model load error: {e}")

    @property
    def is_ready(self):
        return self.scaler is not None and self.features is not None and \
               (self.rf_model is not None or self.nn_model is not None)

    @property
    def active_model_name(self):
        models = []
        if self.rf_model  is not None: models.append("RF")
        if self.gb_model  is not None: models.append("GB")
        if self.xgb_model is not None: models.append("XGB")
        if self.nn_model  is not None: models.append("NN")
        if len(models) > 1: return f"Voting Ensemble ({'+'.join(models)})"
        if self.rf_model  is not None: return "Random Forest"
        return "Neural Network"

    @staticmethod
    def sanitize(text):
        s = re.sub(r"[^a-z0-9_]+","_",str(text).strip().lower())
        return re.sub(r"_+","_",s).strip("_")

    def predict_proba(self, X):
        prob_list = []
        if self.rf_model  is not None:
            prob_list.append(self.rf_model.predict_proba(X)[:,1])
        if self.gb_model  is not None:
            prob_list.append(self.gb_model.predict_proba(X)[:,1])
        if self.xgb_model is not None:
            prob_list.append(self.xgb_model.predict_proba(X)[:,1])
        if self.nn_model  is not None:
            X_sc = self.scaler.transform(X)
            prob_list.append(self.nn_model.predict(X_sc, verbose=0).ravel())
        if not prob_list:
            raise RuntimeError("No models loaded!")
        return np.mean(prob_list, axis=0)

    def predict_risk(self, X):
        prob = float(self.predict_proba(X)[0])
        return prob, prob_to_risk(prob)

    def encode_upload(self, df):
        df_proc = df.copy()
        df_proc.columns = [self.sanitize(c) for c in df_proc.columns]
        req = [self.sanitize(f) for f in self.features]
        val_map = {"yes":1.0,"no":0.0,"1":1.0,"0":0.0,"1.0":1.0,"0.0":0.0,
                   "nan":0.0,"n/a":0.0,"na":0.0}
        for col in df_proc.columns:
            df_proc[col] = df_proc[col].astype(str).str.lower().str.strip().map(val_map).fillna(0.0)
        orig_features = [f for f in req if f not in ['missing_count','risk_density','safety_score']]
        eng_features  = [f for f in req if f in ['missing_count','risk_density','safety_score']]

        missing = [f for f in orig_features if f not in df_proc.columns]
        if missing:
            st.error(f"{len(missing)} required columns missing. Download the template.")
            return None

        X = df_proc[orig_features].copy()

        if 'missing_count' in eng_features:
            X['missing_count'] = df_proc[orig_features].isnull().sum(axis=1).values
        if 'risk_density' in eng_features:
            X['risk_density'] = (X[orig_features] == 0).sum(axis=1)
        if 'safety_score' in eng_features:
            X['safety_score'] = X[orig_features].sum(axis=1) / len(orig_features)

        X = X[[f for f in self.features if f in X.columns]]
        for f in self.features:
            if f not in X.columns:
                X[f] = 0.0
        X = X[self.features]
        return X

    def get_feature_importance(self):
        if self.rf_model and hasattr(self.rf_model,"feature_importances_"):
            return dict(zip(self.features, self.rf_model.feature_importances_))
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# CHATBOT
# ─────────────────────────────────────────────────────────────────────────────
def build_chatbot_context(engine) -> str:
    parts = []
    if "last_results" in st.session_state:
        df = st.session_state["last_results"]
        total = len(df); dist = df["Risk Level"].value_counts().to_dict()
        avg_p = df["Maintenance Probability"].mean()
        needs = (df["Needs Maintenance"]=="Yes").sum() if "Needs Maintenance" in df.columns else 0

        unit_list = ""
        if "Unit Name" in df.columns:
            top = df.sort_values("Maintenance Probability",ascending=False).head(10)
            unit_list = "\n".join([
                f"  - {r['Unit Name']}: {r['Maintenance Probability']:.1f}% ({r['Risk Level']})"
                for _,r in top.iterrows()
            ])

        parts.append(f"""
=== ACTIVE FLEET DATA ===
Total Units: {total}
Average Risk Score: {avg_p:.1f}%
Units Needing Maintenance: {needs}

Risk Breakdown:
- Critical  (75-100%): {dist.get('Critical',0)} units ({dist.get('Critical',0)/total*100:.1f}%)
- Medium    (50-75%):  {dist.get('Medium Risk',0)} units ({dist.get('Medium Risk',0)/total*100:.1f}%)
- Low Risk  (25-50%):  {dist.get('Low Risk',0)} units ({dist.get('Low Risk',0)/total*100:.1f}%)
- Safe      (0-25%):   {dist.get('Safe',0)} units ({dist.get('Safe',0)/total*100:.1f}%)

Top 10 Highest-Risk Units (use these exact names in your answers):
{unit_list if unit_list else 'No unit names available'}

Fleet Status: {"HIGH ALERT — Immediate action required" if avg_p>60 else "MODERATE — Scheduled maintenance advised" if avg_p>35 else "NOMINAL — Within acceptable parameters"}
""")
        imp = engine.get_feature_importance() if engine else {}
        if imp:
            top5 = sorted(imp.items(),key=lambda x:x[1],reverse=True)[:5]
            parts.append("Key Risk Drivers:\n" + "\n".join([f"  - {f}: {v:.3f}" for f,v in top5]))
    else:
        parts.append("NO FLEET DATA LOADED — Instruct user to upload data in Bulk Processor.")

    user = st.session_state.get("user",{})
    parts.append(f"\nSession: {user.get('name','')} ({user.get('role','')}), {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    return "\n".join(parts)

def build_system_prompt(ctx):
    return f"""You are ARIA (Advanced Risk Intelligence Assistant), embedded AI advisor for the Schindler Elevator AI Maintenance Platform.

ROLE: Senior maintenance engineer, 15+ years experience. Precise, professional, safety-first.

CAPABILITIES:
1. Fleet health analysis — cite specific unit names and numbers
2. Prioritized maintenance plans (day-by-day, using actual unit names like "Unit 1", "Unit 5")
3. Root cause and pattern analysis
4. Executive-level reporting
5. Resource-constrained decisions ("which 5 units to service this week")
6. Business impact and risk communication

STRICT RULES:
- ALWAYS use actual unit names from the data (Unit 1, Unit 5, etc.) — never say "these units" or "the first unit"
- Cite specific percentages and counts from the data
- For maintenance plans: list exact unit names per day with specific systems to inspect
- If no data is loaded: guide user to Bulk Processor to upload
- Structure complex answers clearly with numbered lists or sections
- Be concise but complete — executives and engineers both read your output

CURRENT PLATFORM DATA:
{ctx}

You are ARIA. Be direct, specific, and actionable."""

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def render_fleet_charts(df: pd.DataFrame):
    color_map = {k:v["color"] for k,v in RISK_CONFIG.items()}
    total = len(df); avg_p = df["Maintenance Probability"].mean()
    dist  = df["Risk Level"].value_counts()
    crit  = len(df[df["Risk Level"]=="Critical"])
    need  = (df["Needs Maintenance"]=="Yes").sum() if "Needs Maintenance" in df.columns else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Units",      total)
    c2.metric("Critical",         crit,  delta=f"{crit/total*100:.1f}%", delta_color="inverse")
    c3.metric("Need Maintenance", need)
    c4.metric("Safe",             len(df[df["Risk Level"]=="Safe"]))
    c5.metric("Avg Risk Score",   f"{avg_p:.1f}%")
    st.markdown("---")

    col_a, col_b = st.columns(2)
    fig_pie = px.pie(df, names="Risk Level", title="Risk Distribution",
                     color="Risk Level", color_discrete_map=color_map, hole=0.5)
    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#9ba8c9", title_font_size=13)
    col_a.plotly_chart(fig_pie, use_container_width=True)

    fig_hist = px.histogram(df, x="Maintenance Probability", nbins=20,
                            title="Risk Score Distribution", color_discrete_sequence=["#4B9EFF"])
    fig_hist.add_vline(x=50, line_dash="dash", line_color="#b02020",
                       annotation_text="50% alert", annotation_font_color="#b02020")
    fig_hist.add_vline(x=avg_p, line_dash="dot", line_color="#6b7a9f",
                       annotation_text=f"Fleet avg {avg_p:.0f}%", annotation_font_color="#6b7a9f")
    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#9ba8c9", title_font_size=13)
    col_b.plotly_chart(fig_hist, use_container_width=True)

    col_c, col_d = st.columns(2)
    top10 = df.sort_values("Maintenance Probability",ascending=False).head(10).copy()
    fig_top = px.bar(top10, x="Unit Name" if "Unit Name" in top10.columns else top10.index,
                     y="Maintenance Probability", color="Risk Level",
                     color_discrete_map=color_map, title="Top 10 Highest Risk Units",
                     labels={"Unit Name":"","Maintenance Probability":"Risk %"}, height=300)
    fig_top.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#9ba8c9", showlegend=False, title_font_size=13,
                          xaxis_tickangle=-35)
    col_c.plotly_chart(fig_top, use_container_width=True)

    fig_tier = px.bar(x=dist.index, y=dist.values, color=dist.index,
                      color_discrete_map=color_map, title="Units per Risk Tier",
                      labels={"x":"","y":"Count"}, height=300)
    fig_tier.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#9ba8c9", showlegend=False, title_font_size=13)
    col_d.plotly_chart(fig_tier, use_container_width=True)

    sorted_df = df.sort_values("Maintenance Probability",ascending=False).reset_index(drop=True)
    fig_sc = px.scatter(
        sorted_df, x=sorted_df.index, y="Maintenance Probability",
        color="Risk Level", color_discrete_map=color_map,
        hover_name="Unit Name" if "Unit Name" in sorted_df.columns else None,
        title="All Units — Risk Score Map (hover for unit name)",
        labels={"x":"Unit Rank","Maintenance Probability":"Risk %"}, height=270,
    )
    fig_sc.add_hline(y=75, line_dash="dash", line_color="#b02020", annotation_text="Critical")
    fig_sc.add_hline(y=50, line_dash="dot",  line_color="#c06000", annotation_text="Medium")
    fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font_color="#9ba8c9", title_font_size=13)
    st.plotly_chart(fig_sc, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────────────────────────────────────
def build_pdf(df):
    pdf = FPDF(); pdf.add_page(); pdf.set_auto_page_break(True,15)
    pdf.set_font("Arial","B",18); pdf.set_fill_color(13,15,20); pdf.set_text_color(232,234,240)
    pdf.cell(0,14,"Schindler Fleet Maintenance Report",ln=True,fill=True,align="C")
    pdf.set_font("Arial","",9); pdf.set_text_color(107,122,159)
    analyst = st.session_state.get("user",{}).get("name","System")
    pdf.cell(0,8,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Analyst: {analyst}",ln=True,align="C")
    pdf.ln(5); pdf.set_font("Arial","B",13); pdf.set_text_color(30,30,30)
    pdf.cell(0,10,"Executive Summary",ln=True); pdf.set_font("Arial","",10)
    pdf.cell(0,7,f"Total Units: {len(df)}",ln=True)
    if "Risk Level" in df.columns:
        for level in ["Critical","Medium Risk","Low Risk","Safe"]:
            n = len(df[df["Risk Level"]==level])
            pdf.cell(0,7,f"  {level}: {n} ({n/len(df)*100:.1f}%)",ln=True)
    pdf.ln(5); pdf.set_font("Arial","B",12)
    pdf.cell(0,10,"Unit Risk Rankings",ln=True)
    cols = [c for c in ["Unit Name","Risk Level","Maintenance Probability","Needs Maintenance"] if c in df.columns]
    pdf.set_font("Arial","B",9); pdf.set_fill_color(220,220,225)
    cw = 190//len(cols)
    for c in cols: pdf.cell(cw,8,c[:22],border=1,fill=True)
    pdf.ln(); pdf.set_font("Arial","",8)
    for _,row in df.sort_values("Maintenance Probability",ascending=False).head(100).iterrows():
        for c in cols: pdf.cell(cw,6,str(row.get(c,""))[:22],border=1)
        pdf.ln()
    return pdf.output(dest="S").encode("latin-1")

# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────
def render_auth():
    st.markdown("""
    <div class="auth-wrap">
        <div class="auth-monogram"><span>S</span></div>
        <div class="auth-wordmark">SCHINDLER</div>
        <div class="auth-tagline">AI Maintenance Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1,1.1,1])
    with col_c:
        tab_login, tab_register = st.tabs(["Sign In", "Register (Engineers)"])

        with tab_login:
            username = st.text_input("Username", key="li_user", placeholder="Enter username")
            password = st.text_input("Password", type="password", key="li_pw", placeholder="Enter password")
            if st.button("Sign In", use_container_width=True, type="primary", key="li_btn"):
                user = db_get_user(username)
                if user and user["password"] == hash_pw(password):
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user
                    st.session_state["sidebar_visible"] = True
                    db_log_activity(user["id"], "LOGIN")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            st.markdown('<div class="auth-hint">Demo access — <kbd>admin</kbd> / <kbd>admin2026</kbd></div>', unsafe_allow_html=True)

        with tab_register:
            st.markdown('<div style="font-size:0.8rem; color:#6b7a9f; margin-bottom:12px;">Engineer self-registration (max 15 engineers)</div>', unsafe_allow_html=True)
            reg_name = st.text_input("Full Name",        key="reg_name", placeholder="e.g. Ahmed Hassan")
            reg_user = st.text_input("Username",         key="reg_user", placeholder="e.g. engineer1")
            reg_pw   = st.text_input("Password",         type="password", key="reg_pw", placeholder="Min 6 characters")
            reg_pw2  = st.text_input("Confirm Password", type="password", key="reg_pw2")
            if st.button("Register", use_container_width=True, type="primary", key="reg_btn"):
                if not all([reg_name, reg_user, reg_pw]):
                    st.error("All fields required.")
                elif len(reg_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                elif reg_pw != reg_pw2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = db_register_engineer(reg_user, reg_name, reg_pw)
                    st.success(msg) if ok else st.error(msg)

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
class SchindlerApp:
    def __init__(self, engine):
        self.engine = engine
        if "groq_api_key" not in st.session_state:
            st.session_state["groq_api_key"] = os.getenv("GROQ_API_KEY", "")
        if "sidebar_visible" not in st.session_state:
            st.session_state["sidebar_visible"] = True

    # ─── SIDEBAR ─────────────────────────────────────────────────────────────
    def render_sidebar(self) -> str:
        user = st.session_state.get("user", {})
        role = user.get("role", "")
        role_color = "#4b9eff" if role == "admin" else "#2a7f4f"

        engineer_pages = ["Dashboard", "Single Diagnostics", "Bulk Processor", "AI Chatbot"]
        admin_pages    = ["Dashboard", "Bulk Processor", "Admin Panel", "AI Chatbot"]
        pages = admin_pages if role == "admin" else engineer_pages

        # ── If sidebar is hidden: show a floating ☰ button to reopen ──
        if not st.session_state["sidebar_visible"]:
            st.markdown("""
            <style>
            [data-testid="stSidebar"] {
                display: none !important;
                width: 0 !important;
                min-width: 0 !important;
            }
            </style>
            """, unsafe_allow_html=True)

            # Floating open button using Streamlit button pinned via CSS trick
            col_btn, col_rest = st.columns([0.04, 0.96])
            with col_btn:
                if st.button("☰", key="open_sidebar_btn", help="Open navigation menu"):
                    st.session_state["sidebar_visible"] = True
                    st.rerun()

            # Return a default page so the app still renders
            return st.session_state.get("current_page", pages[0])

        # ── Sidebar is visible ──
        with st.sidebar:
            # Top row: logo + close button
            col_logo, col_gap, col_close = st.columns([3, 0.5, 1])
            with col_logo:
                if os.path.exists("Schindler_2.png"):
                    st.image("Schindler_2.png", width=120)
                else:
                    st.markdown(
                        '<div style="font-size:1.1rem;font-weight:700;color:#e8eaf0;'
                        'letter-spacing:0.1em;padding:8px 0;">SCHINDLER</div>',
                        unsafe_allow_html=True
                    )
            with col_close:
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                if st.button(
                    "✕",
                    key="close_sidebar_btn",
                    help="Hide sidebar",
                    use_container_width=True
                ):
                    st.session_state["sidebar_visible"] = False
                    st.rerun()

            # User info card
            st.markdown(f"""
            <div class="side-user-card">
                <div class="u-name">{user.get('name','')}</div>
                <span class="u-role" style="background:{role_color}22; color:{role_color};">
                    {role.upper()}
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            page = st.radio("Navigation", pages, label_visibility="collapsed")
            st.session_state["current_page"] = page
            st.markdown("---")

            st.markdown('<div class="label-sm">System</div>', unsafe_allow_html=True)
            if self.engine.is_ready:
                model_type = self.engine.active_model_name
                threshold_display = f"{self.engine.threshold:.3f}"
                st.markdown(
                    f'<div class="side-status"><span class="dot" style="background:var(--safe);'
                    f'box-shadow:0 0 8px var(--safe);"></span>'
                    f'<span style="color:#bfe8d2;">{model_type} — Active</span></div>', unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="side-status"><span class="dot" style="background:var(--accent);"></span>'
                    f'<span style="color:var(--ink-2);">Threshold {threshold_display} (optimized)</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="side-status"><span class="dot" style="background:var(--crit);'
                    'box-shadow:0 0 8px var(--crit);"></span>'
                    '<span style="color:#f0b6b8;">Model Offline</span></div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")
            with st.expander("API Configuration"):
                st.session_state["groq_api_key"] = st.text_input(
                    "Groq API Key", value=st.session_state["groq_api_key"],
                    type="password", placeholder="gsk_...", label_visibility="collapsed"
                )
                if not st.session_state["groq_api_key"]:
                    st.caption("Required for AI Assistant")

            st.markdown("---")
            if st.button("Sign Out", use_container_width=True):
                uid = st.session_state.get("user", {}).get("id")
                if uid: db_log_activity(uid, "LOGOUT")
                for k in ["authenticated","user","last_results","messages",
                          "single_result","sidebar_visible","current_page"]:
                    st.session_state.pop(k, None)
                st.rerun()

            st.markdown(
                '<div class="side-version">SCHINDLER AI PLATFORM · V6.0</div>',
                unsafe_allow_html=True
            )

        return page

    # ─── DASHBOARD ───────────────────────────────────────────────────────────
    def render_dashboard(self):
        st.markdown('<div class="section-header">Fleet Intelligence Dashboard</div>', unsafe_allow_html=True)

        has_bulk   = "last_results" in st.session_state
        has_single = "single_result" in st.session_state

        if not has_bulk and not has_single:
            st.markdown("""
            <div class="info-card" style="text-align:center; padding:48px; margin-top:32px;">
                <div style="font-size:1rem; color:#6b7a9f; margin-bottom:10px;">No data loaded</div>
                <div style="font-size:0.82rem; color:#3d4a6b;">
                    Run a Single Diagnostic or upload a fleet file in Bulk Processor to populate this dashboard.
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        if has_bulk:
            df = st.session_state["last_results"]
            st.markdown("### Bulk Fleet Analysis")
            render_fleet_charts(df)
            st.markdown("---")
            st.markdown("### All Fleet Records")
            st.dataframe(
                df[["Unit Name","Risk Level","Maintenance Probability","Needs Maintenance"]]
                .sort_values("Maintenance Probability", ascending=False),
                use_container_width=True, height=320
            )

        if has_single:
            st.markdown("---")
            st.markdown("### Last Single Unit Diagnostic")
            res = st.session_state["single_result"]
            risk = res["risk"]; prob = res["probability"]
            c1,c2,c3 = st.columns(3)
            c1.metric("Risk Level",  risk)
            c2.metric("Risk Score",  f"{prob*100:.1f}%")
            c3.metric("Timestamp",   res.get("timestamp","")[:16])
            st.markdown(
                f'<div class="info-card"><div class="label-sm">Recommendation</div>'
                f'<div style="font-size:0.9rem; color:#e8eaf0; margin-top:6px;">'
                f'{RISK_RECS[risk]}</div></div>',
                unsafe_allow_html=True
            )

    # ─── SINGLE DIAGNOSTICS ──────────────────────────────────────────────────
    def render_single(self):
        st.markdown('<div class="section-header">Single Unit Diagnostics</div>', unsafe_allow_html=True)
        if not self.engine.is_ready:
            st.warning("Model offline."); return

        features = self.engine.features
        input_dict = {}
        st.markdown('<div style="font-size:0.82rem; color:#6b7a9f; margin-bottom:16px;">Enter technician checklist responses.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            cols = st.columns(3)
            for i, feat in enumerate(features):
                with cols[i % 3]:
                    label = feat.replace("_", " ").strip()[:55]
                    choice = st.selectbox(label, ["No","Yes","N/A"], key=f"s_{i}")
                    input_dict[feat] = 1.0 if choice == "Yes" else 0.0

        st.markdown("---")
        if st.button("Run Diagnostic", use_container_width=True, type="primary"):
            X    = pd.DataFrame([input_dict])
            prob = float(self.engine.predict_proba(X)[0])
            risk = prob_to_risk(prob); cfg = RISK_CONFIG[risk]

            st.markdown(
                f'<div class="status-banner" style="background:{cfg["bg"]};'
                f'border:1px solid {cfg["color"]}40;color:{cfg["color"]};">'
                f'STATUS: {risk.upper()}&nbsp;&nbsp;|&nbsp;&nbsp;'
                f'<span style="color:#9ba8c9;font-weight:400;">Risk Score: {prob*100:.1f}%</span></div>',
                unsafe_allow_html=True
            )
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                number={"suffix":"%","font":{"size":36,"color":"#e8eaf0"}},
                title={"text":"Failure Risk Score","font":{"color":"#9ba8c9","size":13}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#4b5880"},
                       "bar":{"color":cfg["color"]},"bgcolor":"#13161f","bordercolor":"#1f2230",
                       "steps":[{"range":[0,25],"color":"rgba(42,127,79,0.12)"},
                                 {"range":[25,50],"color":"rgba(184,134,11,0.12)"},
                                 {"range":[50,75],"color":"rgba(192,96,0,0.12)"},
                                 {"range":[75,100],"color":"rgba(176,32,32,0.12)"}]},
            ))
            fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#e0e0e0", margin=dict(l=20,r=20,t=30,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f'<div class="info-card"><div class="label-sm">Recommendation</div>'
                f'<div style="font-size:0.9rem; color:#e8eaf0; margin-top:6px;">'
                f'{RISK_RECS[risk]}</div></div>',
                unsafe_allow_html=True
            )

            result = {"risk":risk, "probability":prob, "timestamp":datetime.now().isoformat()}
            st.session_state["single_result"] = result
            user = st.session_state.get("user", {})
            if user.get("id"):
                db_save_single_diag(user["id"], risk, prob)
                db_log_activity(user["id"], "SINGLE_DIAG", f"{risk} — {prob*100:.1f}%")

    # ─── BULK PROCESSOR ──────────────────────────────────────────────────────
    def render_bulk(self):
        st.markdown('<div class="section-header">Bulk Fleet Processor</div>', unsafe_allow_html=True)
        if not self.engine.is_ready:
            st.warning("Model offline."); return

        with st.expander("Download Inspection Template"):
            tpl = pd.DataFrame(columns=self.engine.features)
            tpl.loc[0] = ["No"] * len(self.engine.features)
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as w: tpl.to_excel(w, index=False)
            st.download_button(
                "Download Template (.xlsx)", buf.getvalue(),
                "Schindler_Template.xlsx", use_container_width=True
            )

        st.markdown("---")
        file = st.file_uploader("Upload Fleet Inspection Database (.xlsx)", type=["xlsx"])
        if file is None: return

        df_orig = pd.read_excel(file)
        st.markdown(
            f'<div style="font-size:0.82rem; color:#6b7a9f; margin:8px 0 16px;">'
            f'{file.name} · {len(df_orig):,} records detected</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns([3,1])
        with c2: threshold = st.slider("Alert Threshold", 0.1, 0.9, 0.5, 0.05)
        with c1: run = st.button("Process Fleet Data", use_container_width=True, type="primary")
        if not run: return

        with st.spinner("Analyzing fleet telemetry..."):
            X = self.engine.encode_upload(df_orig)
            if X is None: return
            probs = self.engine.predict_proba(X)
            df_orig.insert(0, "Unit Name", [f"Unit {i+1}" for i in range(len(df_orig))])
            df_orig["Maintenance Probability"] = np.round(probs * 100, 1)
            df_orig["Risk Level"]              = [prob_to_risk(p) for p in probs]
            df_orig["Needs Maintenance"]       = ["Yes" if p >= threshold else "No" for p in probs]

        st.session_state["last_results"] = df_orig

        user = st.session_state.get("user", {})
        if user.get("id"):
            db_save_upload(user["id"], file.name, df_orig)
            db_log_activity(user["id"], "BULK_UPLOAD", f"{file.name} — {len(df_orig)} units")

        st.markdown(
            f'<div class="bulk-done">'
            f'Analysis complete — {len(df_orig):,} units processed and named '
            f'(Unit 1 to Unit {len(df_orig)})</div>',
            unsafe_allow_html=True
        )

        render_fleet_charts(df_orig)
        st.markdown("---")

        st.markdown("### Unit Rankings — Sorted by Risk Priority")
        display_df = df_orig[["Unit Name","Risk Level","Maintenance Probability","Needs Maintenance"]]\
            .sort_values("Maintenance Probability", ascending=False)
        st.dataframe(display_df, use_container_width=True, height=400)

        c1, c2 = st.columns(2)
        xl = io.BytesIO(); df_orig.to_excel(xl, index=False)
        c1.download_button("Export Results (.xlsx)", xl.getvalue(), "Fleet_Results.xlsx", use_container_width=True)
        c2.download_button("Export Executive Report (.pdf)", build_pdf(df_orig), "Fleet_Report.pdf", use_container_width=True)

    # ─── CHATBOT ─────────────────────────────────────────────────────────────
    def render_chatbot(self):
        st.markdown('<div class="section-header">AI Maintenance Assistant — ARIA</div>', unsafe_allow_html=True)
        if not GROQ_AVAILABLE:
            st.error("Run `pip install langchain-groq`"); return

        groq_key = st.session_state.get("groq_api_key", "")
        if not groq_key:
            st.markdown(
                '<div class="info-card" style="text-align:center;">'
                '<div style="color:#6b7a9f; padding:24px;">'
                'Configure Groq API key in the sidebar to enable ARIA.</div></div>',
                unsafe_allow_html=True
            )
            return

        if "last_results" in st.session_state:
            df = st.session_state["last_results"]
            crit = len(df[df["Risk Level"]=="Critical"])
            st.markdown(
                f'<div class="ctx-chip ok">Fleet context active — {len(df)} units '
                f'({crit} critical) — ARIA has full data access including unit names</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="ctx-chip off">'
                'No fleet data — upload in Bulk Processor for full analysis capability.</div>',
                unsafe_allow_html=True
            )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if len(st.session_state.messages) == 0:
            st.markdown('<div class="label-sm" style="margin-bottom:8px;">Suggested queries</div>', unsafe_allow_html=True)
            sugg = [
                "Give me a full fleet health summary",
                "Which units need immediate maintenance?",
                "Build a 7-day maintenance schedule",
                "Write an executive report for management",
                "If I can only service 5 units this week, which ones?",
                "What are the business risks if no action is taken?",
            ]
            cols = st.columns(3)
            for i, s in enumerate(sugg):
                if cols[i % 3].button(s, key=f"sug_{i}", use_container_width=True):
                    st.session_state.messages.append({"role":"user","content":s})
                    st.rerun()

        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        prompt = st.chat_input("Ask ARIA about fleet status, maintenance planning, or risk analysis...")
        if not prompt: return

        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)

        ctx   = build_chatbot_context(self.engine)
        sys_p = build_system_prompt(ctx)
        try:
            with st.spinner("ARIA is analyzing..."):
                llm = ChatGroq(temperature=0.2, groq_api_key=groq_key, model_name="llama-3.3-70b-versatile")
                full_msgs = [{"role":"system","content":sys_p}] + \
                            [{"role":m["role"],"content":m["content"]} for m in st.session_state.messages]
                response = llm.invoke(full_msgs).content
            with st.chat_message("assistant"): st.markdown(response)
            st.session_state.messages.append({"role":"assistant","content":response})
            uid = st.session_state.get("user",{}).get("id")
            if uid: db_log_activity(uid, "CHATBOT", prompt[:80])
        except Exception as e:
            st.error(f"Communication error: {e}")

        col_l, col_r = st.columns([5,1])
        with col_r:
            if st.button("Clear History", use_container_width=True):
                st.session_state.messages = []; st.rerun()

    # ─── ADMIN PANEL ─────────────────────────────────────────────────────────
    def render_admin(self):
        color_map = {k:v["color"] for k,v in RISK_CONFIG.items()}
        engineers = db_get_all_engineers()
        stats     = db_get_platform_stats()

        st.markdown("""
        <div style="margin-bottom:28px;">
            <div class="page-kicker">Admin Command Center</div>
            <div class="page-title">Fleet Intelligence — Platform Administration</div>
            <div class="page-sub">Full platform visibility · All engineers · All sessions</div>
        </div>
        """, unsafe_allow_html=True)

        total_units_analyzed = 0
        total_critical = 0
        for eng in engineers:
            for up in db_get_engineer_uploads(eng["id"]):
                total_units_analyzed += up.get("total_units", 0)
                total_critical       += up.get("critical", 0)

        k1,k2,k3,k4,k5,k6 = st.columns(6)
        k1.metric("Engineers",        f"{len(engineers)}/15")
        k2.metric("Total Uploads",    stats["total_uploads"])
        k3.metric("Single Diags",     stats["total_diags"])
        k4.metric("Total Logins",     stats["total_logins"])
        k5.metric("Units Analyzed",   f"{total_units_analyzed:,}")
        k6.metric("Total Critical",   total_critical,
                  delta=f"{total_critical/max(total_units_analyzed,1)*100:.1f}%",
                  delta_color="inverse")

        st.markdown("---")

        # ── Platform Overview Charts ──
        if engineers and stats["total_uploads"] > 0:
            with st.expander("📊  Platform Overview — All Engineers Combined", expanded=True):
                all_uploads = []
                for eng in engineers:
                    for up in db_get_engineer_uploads(eng["id"]):
                        up["engineer_name"] = eng["name"]
                        all_uploads.append(up)

                if all_uploads:
                    up_summary = pd.DataFrame(all_uploads)

                    ca, cb = st.columns(2)

                    eng_counts = up_summary.groupby("engineer_name")["total_units"].sum().reset_index()
                    eng_counts.columns = ["Engineer","Units Analyzed"]
                    fig_eng = px.bar(eng_counts, x="Engineer", y="Units Analyzed",
                                     title="Total Units Analyzed per Engineer",
                                     color_discrete_sequence=["#4B9EFF"], height=280)
                    fig_eng.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font_color="#9ba8c9", title_font_size=12, showlegend=False)
                    ca.plotly_chart(fig_eng, use_container_width=True)

                    total_crit = up_summary["critical"].sum()
                    total_med  = up_summary["medium"].sum()
                    total_low  = up_summary["low_risk"].sum()
                    total_safe = up_summary["safe"].sum()
                    pie_data   = pd.DataFrame({
                        "Risk Level": ["Critical","Medium Risk","Low Risk","Safe"],
                        "Count":      [total_crit, total_med, total_low, total_safe]
                    })
                    fig_pie = px.pie(pie_data, names="Risk Level", values="Count",
                                     title="Platform-Wide Risk Distribution",
                                     color="Risk Level", color_discrete_map=color_map,
                                     hole=0.5, height=280)
                    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#9ba8c9", title_font_size=12)
                    cb.plotly_chart(fig_pie, use_container_width=True)

                    up_summary["timestamp_short"] = up_summary["timestamp"].str[:16]
                    up_summary_sorted = up_summary.sort_values("timestamp")
                    fig_trend = px.line(up_summary_sorted, x="timestamp_short", y="avg_risk",
                                        color="engineer_name", markers=True,
                                        title="Avg Risk Score per Upload Session (All Engineers)",
                                        labels={"timestamp_short":"Session","avg_risk":"Avg Risk %",
                                                "engineer_name":"Engineer"},
                                        height=250)
                    fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                            font_color="#9ba8c9", title_font_size=12)
                    st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("---")

        # ── Engineer Deep Dive ──
        st.markdown("""
        <div style="font-size:0.72rem; color:#6b7a9f; letter-spacing:0.12em;
                    text-transform:uppercase; margin-bottom:14px;">
            Engineer Deep Dive
        </div>
        """, unsafe_allow_html=True)

        if not engineers:
            st.markdown('<div class="info-card" style="text-align:center; padding:40px; color:#4b5880;">No engineers registered yet.</div>', unsafe_allow_html=True)
            return

        eng_map = {e["id"]: e for e in engineers}
        selected_id = st.selectbox(
            "Select Engineer",
            options=[e["id"] for e in engineers],
            format_func=lambda x: f"{eng_map[x]['name']}  (@{eng_map[x]['username']})",
            label_visibility="collapsed",
        )
        selected = eng_map[selected_id]
        uploads  = db_get_engineer_uploads(selected_id)
        diags    = db_get_engineer_single_diags(selected_id)
        acts     = db_get_engineer_activity(selected_id)

        eng_total_units = sum(u.get("total_units",0) for u in uploads)
        eng_total_crit  = sum(u.get("critical",0)    for u in uploads)
        eng_avg_risk    = (sum(u.get("avg_risk",0) for u in uploads) / max(len(uploads),1))

        st.markdown(f"""
        <div class="admin-profile">
            <div>
                <div style="font-size:1.05rem; color:#e8ecf3; font-weight:600;">{selected['name']}</div>
                <div style="font-size:0.78rem; color:#a3aec0; margin-top:3px;">@{selected['username']}</div>
                <div style="font-size:0.72rem; color:#64718a; margin-top:6px;">
                    Registered: {selected['created'][:16]}
                </div>
            </div>
            <div class="admin-stat">
                <div>
                    <div style="font-size:0.68rem; color:#64718a; text-transform:uppercase; letter-spacing:0.1em;">Bulk Uploads</div>
                    <div style="font-size:1.5rem; color:#5b97ff; font-family:'IBM Plex Mono',monospace;">{len(uploads)}</div>
                </div>
                <div>
                    <div style="font-size:0.68rem; color:#64718a; text-transform:uppercase; letter-spacing:0.1em;">Units Analyzed</div>
                    <div style="font-size:1.5rem; color:#e8ecf3; font-family:'IBM Plex Mono',monospace;">{eng_total_units:,}</div>
                </div>
                <div>
                    <div style="font-size:0.68rem; color:#64718a; text-transform:uppercase; letter-spacing:0.1em;">Critical Units</div>
                    <div style="font-size:1.5rem; color:#e5484d; font-family:'IBM Plex Mono',monospace;">{eng_total_crit}</div>
                </div>
                <div>
                    <div style="font-size:0.68rem; color:#64718a; text-transform:uppercase; letter-spacing:0.1em;">Avg Risk Score</div>
                    <div style="font-size:1.5rem; color:#ec7f37; font-family:'IBM Plex Mono',monospace;">{eng_avg_risk:.1f}%</div>
                </div>
                <div>
                    <div style="font-size:0.68rem; color:#64718a; text-transform:uppercase; letter-spacing:0.1em;">Single Diags</div>
                    <div style="font-size:1.5rem; color:#e8ecf3; font-family:'IBM Plex Mono',monospace;">{len(diags)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_up, tab_diag, tab_act = st.tabs([
            f"📁  Bulk Uploads ({len(uploads)})",
            f"🔬  Single Diagnostics ({len(diags)})",
            f"📋  Activity Log ({len(acts)})"
        ])

        # ── TAB 1: BULK UPLOADS ──
        with tab_up:
            if not uploads:
                st.markdown('<div class="info-card" style="text-align:center; padding:32px; color:#4b5880;">No bulk uploads yet for this engineer.</div>', unsafe_allow_html=True)
            else:
                up_df = pd.DataFrame(uploads)[["filename","total_units","critical","medium","low_risk","safe","avg_risk","timestamp"]]
                up_df.columns = ["File","Total","Critical","Medium","Low Risk","Safe","Avg Risk %","Timestamp"]
                up_df["Avg Risk %"] = up_df["Avg Risk %"].round(1)
                up_df["Timestamp"]  = up_df["Timestamp"].str[:16]
                st.dataframe(up_df, use_container_width=True, height=200)

                st.markdown("---")
                st.markdown('<div style="font-size:0.72rem; color:#6b7a9f; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px;">Inspect Specific Upload — Full Analysis</div>', unsafe_allow_html=True)

                sel_idx = st.selectbox(
                    "pick upload",
                    range(len(uploads)),
                    format_func=lambda i: (
                        f"{'🔴' if uploads[i]['critical']>0 else '🟡' if uploads[i]['medium']>0 else '🟢'}  "
                        f"{uploads[i]['filename']}  ·  {uploads[i]['timestamp'][:16]}  ·  "
                        f"{uploads[i]['total_units']} units  ·  "
                        f"{uploads[i]['critical']} critical"
                    ),
                    label_visibility="collapsed",
                    key="admin_upload_sel"
                )
                sel_up = uploads[sel_idx]

                if sel_up.get("result_json"):
                    try:
                        res_df = pd.read_json(io.StringIO(sel_up["result_json"]))
                    except Exception:
                        res_df = pd.DataFrame()

                    if not res_df.empty:
                        if "Needs Maintenance" not in res_df.columns:
                            res_df["Needs Maintenance"] = res_df["Maintenance Probability"].apply(
                                lambda x: "Yes" if x >= 50 else "No"
                            )

                        dist  = res_df["Risk Level"].value_counts().to_dict()
                        avg_p = res_df["Maintenance Probability"].mean()
                        alert_color = "#b02020" if dist.get("Critical",0)>0 else "#c06000" if dist.get("Medium Risk",0)>0 else "#2a7f4f"
                        st.markdown(f"""
                        <div class="session-strip">
                            <div>
                                <div style="font-size:0.68rem; color:#6b7a9f; text-transform:uppercase;">File</div>
                                <div style="color:#e8eaf0; font-size:0.9rem; font-weight:500;">{sel_up['filename']}</div>
                            </div>
                            <div>
                                <div style="font-size:0.68rem; color:#6b7a9f; text-transform:uppercase;">Session</div>
                                <div style="color:#9ba8c9; font-size:0.85rem;">{sel_up['timestamp'][:16]}</div>
                            </div>
                            <div>
                                <div style="font-size:0.68rem; color:#6b7a9f; text-transform:uppercase;">Total Units</div>
                                <div style="color:#4b9eff; font-size:1.1rem; font-family:'IBM Plex Mono',monospace;">{sel_up['total_units']}</div>
                            </div>
                            <div>
                                <div style="font-size:0.68rem; color:#6b7a9f; text-transform:uppercase;">Avg Risk</div>
                                <div style="color:{alert_color}; font-size:1.1rem; font-family:'IBM Plex Mono',monospace;">{avg_p:.1f}%</div>
                            </div>
                            <div>
                                <div style="font-size:0.68rem; color:#6b7a9f; text-transform:uppercase;">Critical</div>
                                <div style="color:#b02020; font-size:1.1rem; font-family:'IBM Plex Mono',monospace;">{dist.get('Critical',0)}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        render_fleet_charts(res_df)

                        st.markdown("---")
                        st.markdown('<div style="font-size:0.72rem; color:#6b7a9f; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;">Unit Rankings</div>', unsafe_allow_html=True)
                        st.dataframe(
                            res_df[["Unit Name","Risk Level","Maintenance Probability","Needs Maintenance"]]
                            .sort_values("Maintenance Probability", ascending=False),
                            use_container_width=True, height=350
                        )

                        xl = io.BytesIO()
                        res_df.to_excel(xl, index=False)
                        st.download_button(
                            f"⬇️  Export {sel_up['filename']} Results (.xlsx)",
                            xl.getvalue(),
                            f"admin_export_{sel_up['filename']}",
                            use_container_width=True
                        )
                    else:
                        st.info("No detailed unit data stored for this upload.")

        # ── TAB 2: SINGLE DIAGNOSTICS ──
        with tab_diag:
            if not diags:
                st.markdown('<div class="info-card" style="text-align:center; padding:32px; color:#4b5880;">No single diagnostics run yet.</div>', unsafe_allow_html=True)
            else:
                diag_df = pd.DataFrame(diags)[["risk_level","probability","timestamp"]]
                diag_df.columns = ["Risk Level","Probability (%)","Timestamp"]
                diag_df["Probability (%)"] = (diag_df["Probability (%)"]*100).round(1)
                diag_df["Timestamp"] = diag_df["Timestamp"].str[:16]

                d1,d2,d3,d4 = st.columns(4)
                d1.metric("Total Diagnostics", len(diag_df))
                d2.metric("Critical Results",  len(diag_df[diag_df["Risk Level"]=="Critical"]))
                d3.metric("Avg Risk Score",    f"{diag_df['Probability (%)'].mean():.1f}%")
                d4.metric("Safe Results",      len(diag_df[diag_df["Risk Level"]=="Safe"]))
                st.markdown("---")

                c1, c2 = st.columns([2,1])
                fig_d = px.bar(diag_df, x="Timestamp", y="Probability (%)", color="Risk Level",
                               color_discrete_map=color_map,
                               title=f"Diagnostic History — {selected['name']}",
                               labels={"Probability (%)":"Risk %"}, height=300)
                fig_d.add_hline(y=75, line_dash="dash", line_color="#b02020", annotation_text="Critical")
                fig_d.add_hline(y=50, line_dash="dot",  line_color="#c06000", annotation_text="Medium")
                fig_d.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font_color="#9ba8c9", title_font_size=12)
                c1.plotly_chart(fig_d, use_container_width=True)

                risk_counts = diag_df["Risk Level"].value_counts().reset_index()
                risk_counts.columns = ["Risk Level","Count"]
                fig_dp = px.pie(risk_counts, names="Risk Level", values="Count",
                                color="Risk Level", color_discrete_map=color_map,
                                hole=0.55, height=300, title="Risk Distribution")
                fig_dp.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#9ba8c9", title_font_size=12)
                c2.plotly_chart(fig_dp, use_container_width=True)

                st.dataframe(diag_df, use_container_width=True, height=250)

        # ── TAB 3: ACTIVITY LOG ──
        with tab_act:
            if not acts:
                st.markdown('<div class="info-card" style="text-align:center; padding:32px; color:#4b5880;">No activity recorded.</div>', unsafe_allow_html=True)
            else:
                act_df = pd.DataFrame(acts)[["action","detail","timestamp"]]
                act_df.columns = ["Action","Detail","Timestamp"]
                act_df["Timestamp"] = act_df["Timestamp"].str[:16]

                action_counts = act_df["Action"].value_counts().reset_index()
                action_counts.columns = ["Action","Count"]
                fig_act = px.bar(action_counts, x="Action", y="Count",
                                 title="Activity Breakdown", color_discrete_sequence=["#4B9EFF"],
                                 height=220)
                fig_act.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font_color="#9ba8c9", title_font_size=12, showlegend=False)
                st.plotly_chart(fig_act, use_container_width=True)
                st.dataframe(act_df, use_container_width=True, height=300)

        # ── All Engineers Table ──
        st.markdown("---")
        st.markdown('<div style="font-size:0.72rem; color:#6b7a9f; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:12px;">All Registered Engineers</div>', unsafe_allow_html=True)

        eng_rows = []
        for e in engineers:
            ups = db_get_engineer_uploads(e["id"])
            dgs = db_get_engineer_single_diags(e["id"])
            eng_rows.append({
                "Name":          e["name"],
                "Username":      f"@{e['username']}",
                "Registered":    e["created"][:16],
                "Uploads":       len(ups),
                "Units Analyzed":sum(u.get("total_units",0) for u in ups),
                "Critical Units":sum(u.get("critical",0) for u in ups),
                "Single Diags":  len(dgs),
                "Avg Risk %":    round(sum(u.get("avg_risk",0) for u in ups)/max(len(ups),1),1),
            })
        if eng_rows:
            st.dataframe(pd.DataFrame(eng_rows), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing prediction engine...")
def load_engine():
    return ElevatorRiskModel()

def main():
    init_db()

    if not st.session_state.get("authenticated"):
        render_auth()
        return

    engine = load_engine()
    app    = SchindlerApp(engine)
    page   = app.render_sidebar()

    page_map = {
        "Dashboard":          app.render_dashboard,
        "Single Diagnostics": app.render_single,
        "Bulk Processor":     app.render_bulk,
        "AI Chatbot":         app.render_chatbot,
        "Admin Panel":        app.render_admin,
    }
    if page in page_map:
        page_map[page]()

if __name__ == "__main__":
    main()