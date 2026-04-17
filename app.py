"""
╔══════════════════════════════════════════════════════════════════╗
║                    QueryDoc  v4.0                                   ║
║        Advanced Document Intelligence & Deep Analysis            ║
╚══════════════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
import gc
from pathlib import Path

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

import re, json, time, hashlib, tempfile, string
from datetime import datetime
from collections import Counter
import numpy as np

import fitz
from groq import Groq
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px

try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    pytesseract = None
    convert_from_bytes = None
    Image = None


st.set_page_config(
    page_title="QueryDoc",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════
# HIDE STREAMLIT DEFAULT UI + INJECT GLOBAL CSS
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600&display=swap');

/* ── RESET STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] { display: none !important; }
.viewerBadge_container, [data-testid="stViewerBadge"] { display: none !important; }

/* ── GLOBAL TOKENS ── */
:root {
    --ink:         #0a0f1e;
    --ink-2:       #2d3a55;
    --ink-3:       #6b7a99;
    --ink-4:       #a8b4cc;
    --canvas:      #f4f6fb;
    --surface:     #ffffff;
    --surface-2:   rgba(255,255,255,0.75);
    --surface-3:   rgba(255,255,255,0.45);
    --glass-blur:  blur(24px) saturate(180%);
    --border:      rgba(160,185,230,0.30);
    --border-mid:  rgba(120,155,210,0.45);
    --border-hi:   rgba(80,120,200,0.60);
    --sapphire:    #2563eb;
    --sapphire-d:  #1d4ed8;
    --sapphire-l:  #3b82f6;
    --sapphire-gl: rgba(59,130,246,0.18);
    --sapphire-gx: rgba(37,99,235,0.35);
    --emerald:     #059669;
    --amber:       #d97706;
    --ruby:        #dc2626;
    --violet:      #7c3aed;
    --r-sm:  8px;
    --r-md:  14px;
    --r-lg:  20px;
    --r-xl:  28px;
    --r-pill: 999px;
    --sh-xs: 0 1px 4px rgba(10,15,40,0.06), 0 1px 2px rgba(10,15,40,0.04);
    --sh-sm: 0 3px 12px rgba(10,15,40,0.08), 0 1px 4px rgba(10,15,40,0.05);
    --sh-md: 0 8px 28px rgba(10,15,40,0.12), 0 2px 8px rgba(10,15,40,0.06);
    --sh-blue: 0 4px 20px rgba(37,99,235,0.30);
    --sh-ring: 0 0 0 3px rgba(59,130,246,0.20);
}

html, body, [class*="css"] {
    font-family: 'Outfit', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
    color: var(--ink);
}

/* ── CANVAS BACKGROUND ── */
.stApp {
    background: var(--canvas);
    background-image:
        radial-gradient(ellipse 90% 55% at 5% -5%,  rgba(186,220,255,0.35) 0%, transparent 62%),
        radial-gradient(ellipse 70% 50% at 95% 105%, rgba(196,181,253,0.20) 0%, transparent 58%),
        radial-gradient(ellipse 50% 40% at 50% 50%,  rgba(224,231,255,0.12) 0%, transparent 70%);
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(248,251,255,0.88) 100%) !important;
    backdrop-filter: var(--glass-blur) !important;
    -webkit-backdrop-filter: var(--glass-blur) !important;
    border-right: 1px solid var(--border-mid) !important;
    box-shadow: 4px 0 32px rgba(10,15,40,0.08) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* ── SIDEBAR ACTIVE BUTTON ── */
[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--sapphire-l) 0%, var(--sapphire-d) 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--r-md) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.01em !important;
    box-shadow: var(--sh-blue) !important;
    text-align: left !important;
    padding-left: 14px !important;
}
[data-testid="stSidebar"] button[kind="secondary"] {
    background: transparent !important;
    color: var(--ink-2) !important;
    border: 1px solid transparent !important;
    border-radius: var(--r-md) !important;
    font-size: 13px !important;
    text-align: left !important;
    padding-left: 14px !important;
    transition: all 0.16s ease !important;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: rgba(59,130,246,0.08) !important;
    color: var(--sapphire) !important;
    border-color: rgba(59,130,246,0.22) !important;
}
[data-testid="stSidebar"] [data-testid="column"]:nth-of-type(2) button {
    background: transparent !important;
    color: var(--ink-4) !important;
    border: none !important;
    font-size: 11px !important;
    border-radius: var(--r-sm) !important;
    transition: all 0.15s ease !important;
    padding: 0 !important;
    min-height: 36px;
}
[data-testid="stSidebar"] [data-testid="column"]:nth-of-type(2) button:hover {
    color: var(--ruby) !important;
    background: rgba(220,38,38,0.08) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.60);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-mid);
    border-radius: var(--r-lg);
    gap: 3px;
    padding: 5px;
    margin-bottom: 24px;
    box-shadow: var(--sh-xs), inset 0 1px 0 rgba(255,255,255,0.80);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--ink-3);
    border-radius: var(--r-md);
    font-weight: 500;
    font-size: 12px;
    padding: 8px 20px;
    border: none;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    transition: all 0.16s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--sapphire);
    background: rgba(59,130,246,0.07);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--sapphire-l) 0%, var(--sapphire-d) 100%) !important;
    color: #fff !important;
    box-shadow: var(--sh-blue) !important;
}

/* ── GENERIC INPUTS ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.90) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--r-md) !important;
    color: var(--ink) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
    box-shadow: inset 0 1px 3px rgba(10,15,40,0.04) !important;
    transition: border-color 0.16s, box-shadow 0.16s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--sapphire-l) !important;
    box-shadow: var(--sh-ring), inset 0 1px 3px rgba(10,15,40,0.03) !important;
    outline: none !important;
}

/* ── GENERIC BUTTONS ── */
.stButton > button {
    font-family: 'Outfit', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    border-radius: var(--r-md) !important;
    letter-spacing: 0.03em !important;
    transition: all 0.16s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--sapphire-l) 0%, var(--sapphire-d) 100%) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: var(--sh-blue) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 24px rgba(37,99,235,0.42) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.82) !important;
    color: var(--ink-2) !important;
    border: 1px solid var(--border-mid) !important;
    box-shadow: var(--sh-xs) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #fff !important;
    box-shadow: var(--sh-sm) !important;
}

/* ── SELECTS / MULTISELECT ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: rgba(255,255,255,0.90) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--r-md) !important;
    font-size: 13.5px !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(59,130,246,0.38) !important;
    border-radius: var(--r-lg) !important;
    background: rgba(219,234,254,0.12) !important;
    transition: all 0.20s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(59,130,246,0.65) !important;
    background: rgba(219,234,254,0.26) !important;
}

/* Blue browse file button */
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--r-md) !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    box-shadow: var(--sh-blue) !important;
}
[data-testid="stFileUploader"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(37,99,235,0.42) !important;
}

/* ── TOAST / ALERT ── */
[data-testid="stToast"] {
    background: rgba(255,255,255,0.95) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--r-lg) !important;
    box-shadow: var(--sh-md) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stAlert"] {
    border-radius: var(--r-md) !important;
    font-size: 13px !important;
}

/* ── HR ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 14px 0 !important; }

/* ── RADIO ── */
[data-testid="stRadio"] label { font-size: 13px !important; color: var(--ink-2) !important; }


/* ═══════════════════════════════════════════════════════════
   CHAT LAYOUT — FULL-HEIGHT FIXED BOTTOM BAR
   ═══════════════════════════════════════════════════════════ */

/* Main content area needs padding at bottom so messages aren't hidden */
.main .block-container {
    padding-bottom: 0 !important;
    padding-top: 1rem !important;
}

/* The chat scroll area */
.chat-scroll-area {
    height: calc(85vh - 180px);
    overflow-y: auto;
    padding: 8px 4px 24px 4px;
    scroll-behavior: smooth;
}
.chat-scroll-area::-webkit-scrollbar { width: 4px; }
.chat-scroll-area::-webkit-scrollbar-track { background: transparent; }
.chat-scroll-area::-webkit-scrollbar-thumb {
    background: rgba(100,130,200,0.25);
    border-radius: 999px;
}

/* Chat bubbles */
.chat-wrap-user {
    display: flex;
    justify-content: flex-end;
    margin: 10px 0;
}
.chat-wrap-ai {
    display: flex;
    justify-content: flex-start;
    margin: 10px 0;
}
.bubble-user {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: #ffffff;
    border-radius: 20px 20px 5px 20px;
    padding: 12px 18px;
    max-width: 72%;
    font-size: 14px;
    line-height: 1.65;
    box-shadow: 0 4px 16px rgba(37,99,235,0.30);
    letter-spacing: 0.01em;
}
.bubble-ai {
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(160,185,230,0.35);
    border-radius: 20px 20px 20px 5px;
    padding: 12px 18px;
    max-width: 72%;
    font-size: 14px;
    line-height: 1.70;
    color: var(--ink);
    box-shadow: 0 2px 10px rgba(10,15,40,0.07);
}
.avatar-ai {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #3b82f6, #7c3aed);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-size: 13px;
    font-weight: 700;
    flex-shrink: 0;
    margin-right: 10px;
    box-shadow: 0 2px 8px rgba(59,130,246,0.30);
}

/* ── FIXED CHAT INPUT BAR WITH BUTTON INSIDE ── */
.chat-input-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 999;
    background: rgba(244,246,251,0.92);
    backdrop-filter: blur(28px) saturate(200%);
    -webkit-backdrop-filter: blur(28px) saturate(200%);
    border-top: 1px solid rgba(160,185,230,0.40);
    box-shadow: 0 -8px 32px rgba(10,15,40,0.10);
    padding: 14px 24px 18px 24px;
}

/* Push the chat form to not obscure sidebar */
.chat-input-inner {
    max-width: 860px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.90);
    border: 1px solid rgba(120,155,210,0.45);
    border-radius: 999px;
    padding: 4px 4px 4px 20px;
    box-shadow: 0 4px 20px rgba(10,15,40,0.10), 0 0 0 0px rgba(59,130,246,0);
    transition: box-shadow 0.20s ease, border-color 0.20s ease;
}
.chat-input-inner:focus-within {
    border-color: rgba(59,130,246,0.55);
    box-shadow: 0 4px 24px rgba(10,15,40,0.12), 0 0 0 3px rgba(59,130,246,0.15);
}

/* Override streamlit input inside the bar */
.chat-input-bar .stTextInput {
    flex: 1;
}
.chat-input-bar .stTextInput > div > div > input {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    font-size: 14.5px !important;
    padding: 12px 4px !important;
    color: var(--ink) !important;
}
.chat-input-bar .stTextInput > div > div > input:focus {
    box-shadow: none !important;
    border: none !important;
}
.chat-input-bar .stTextInput > div { border: none !important; }

/* Send button inside chat bar */
.chat-send-btn {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}
.chat-send-btn button {
    width: 38px !important;
    height: 38px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
    font-size: 16px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-width: unset !important;
    min-height: unset !important;
    margin: 4px !important;
}
.chat-send-btn button:hover {
    box-shadow: 0 3px 12px rgba(37,99,235,0.45) !important;
    transform: translateY(-1px) !important;
}


/* ═══════════════════════════════════════
   SIDEBAR HEADER / LOGO
   ═══════════════════════════════════════ */
.qd-logo {
    font-family: 'Playfair Display', serif;
    font-size: 26px;
    font-weight: 600;
    background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 50%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    line-height: 1;
}
.qd-tagline {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink-4);
    margin-top: 3px;
}
.qd-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10b981;
    display: inline-block;
    margin-right: 5px;
    box-shadow: 0 0 0 2px rgba(16,185,129,0.25);
    animation: pulse-dot 2.5s infinite;
}
@keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 0 2px rgba(16,185,129,0.25); }
    50% { box-shadow: 0 0 0 5px rgba(16,185,129,0.08); }
}

/* ═══════════════════════════════════════
   CONTENT CARDS
   ═══════════════════════════════════════ */
.qd-card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px) saturate(160%);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    padding: 22px 26px;
    margin: 10px 0;
    box-shadow: var(--sh-sm), inset 0 1px 0 rgba(255,255,255,0.80);
    transition: box-shadow 0.20s ease, transform 0.20s ease;
}
.qd-card:hover {
    box-shadow: var(--sh-md);
    transform: translateY(-1px);
}
.qd-card-accent {
    background: linear-gradient(135deg, rgba(219,234,254,0.65) 0%, rgba(224,231,255,0.45) 100%);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(147,197,253,0.50);
    border-left: 3px solid var(--sapphire-l);
    border-radius: var(--r-lg);
    padding: 20px 24px;
    margin: 10px 0;
    box-shadow: var(--sh-xs), inset 0 1px 0 rgba(255,255,255,0.70);
}
.info-item {
    border-bottom: 1px solid var(--border);
    padding: 14px 0;
    margin: 0;
}
.info-item:last-child {
    border-bottom: none;
}
.info-label {
    font-size: 9.5px;
    font-weight: 700;
    color: var(--ink-4);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
}
.info-value {
    font-size: 14px;
    line-height: 1.65;
    color: var(--ink-2);
}
.section-label {
    font-size: 9.5px;
    font-weight: 700;
    color: var(--sapphire);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin: 18px 0 9px 0;
    display: flex;
    align-items: center;
    gap: 9px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(59,130,246,0.28) 0%, transparent 100%);
}

.rewrite-output {
    background: rgba(255,255,255,0.85);
    border: 1px solid var(--border-mid);
    border-radius: var(--r-lg);
    padding: 26px 28px;
    white-space: pre-wrap;
    margin-top: 18px;
    font-size: 14px;
    line-height: 1.80;
    color: var(--ink);
    box-shadow: var(--sh-sm), inset 0 1px 0 rgba(255,255,255,0.90);
}

/* ═══════════════════════════════════════
   METRIC CARDS
   ═══════════════════════════════════════ */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.metric-box {
    background: rgba(255,255,255,0.82);
    border: 1px solid var(--border);
    border-radius: var(--r-md);
    padding: 16px;
    text-align: center;
    box-shadow: var(--sh-xs);
    backdrop-filter: blur(10px);
}
.metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 28px;
    font-weight: 500;
    color: var(--sapphire);
    letter-spacing: -0.04em;
}
.metric-lbl {
    font-size: 9.5px;
    font-weight: 700;
    color: var(--ink-4);
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-top: 3px;
}

/* ═══════════════════════════════════════
   EMPTY STATE
   ═══════════════════════════════════════ */
.empty-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 70vh;
    text-align: center;
    padding: 40px 20px;
}
.empty-logo {
    font-family: 'Playfair Display', serif;
    font-size: 64px;
    font-weight: 600;
    background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 45%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 14px;
}
.empty-sub {
    color: var(--ink-3);
    font-size: 15px;
    line-height: 1.75;
    max-width: 460px;
    margin: 0 auto 28px auto;
}
.pill {
    display: inline-block;
    background: rgba(255,255,255,0.85);
    border: 1px solid var(--border-mid);
    border-radius: var(--r-pill);
    padding: 6px 18px;
    font-size: 12px;
    font-weight: 600;
    color: var(--sapphire);
    margin: 4px 3px;
    letter-spacing: 0.03em;
    box-shadow: var(--sh-xs);
    backdrop-filter: blur(8px);
}

/* ═══════════════════════════════════════
   THINKING INDICATOR
   ═══════════════════════════════════════ */
.thinking-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 10px 0;
}
.thinking-dots {
    display: flex;
    gap: 5px;
    align-items: center;
}
.thinking-dots span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--sapphire-l);
    animation: dot-bounce 1.4s infinite ease-in-out;
    opacity: 0.7;
}
.thinking-dots span:nth-child(1) { animation-delay: 0s; }
.thinking-dots span:nth-child(2) { animation-delay: 0.20s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.40s; }
@keyframes dot-bounce {
    0%, 80%, 100% { transform: scale(0.75); opacity: 0.5; }
    40% { transform: scale(1.15); opacity: 1; }
}
.thinking-text {
    font-size: 12.5px;
    color: var(--ink-3);
    font-style: italic;
    letter-spacing: 0.01em;
}

/* Sidebar upload label */
.upload-label {
    font-size: 9.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    color: var(--ink-4);
    margin-bottom: 8px;
    display: block;
}
.doc-list-label {
    font-size: 9.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    color: var(--ink-4);
    margin-bottom: 8px;
    display: block;
}

/* Professional extraction styles */
.extraction-section {
    background: rgba(255,255,255,0.80);
    border: 1px solid var(--border);
    border-radius: var(--r-md);
    padding: 18px 20px;
    margin-bottom: 16px;
}
.extraction-title {
    font-size: 11px;
    font-weight: 700;
    color: var(--ink-4);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 12px;
}
.extraction-content {
    font-size: 14px;
    line-height: 1.65;
    color: var(--ink-2);
}
.extraction-list {
    margin: 0;
    padding-left: 20px;
}
.extraction-list li {
    margin: 8px 0;
    line-height: 1.65;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "pdf_store":     {},
        "env_log":       [],
        "last_sent_msg": "",
        "active_pdf":    None,
        "is_thinking":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ════════════════════════════════════════════════════════════════
# DEFERRED HEAVY RESOURCES
# ════════════════════════════════════════════════════════════════
def get_embed_model():
    if "embed_model" not in st.session_state:
        with st.spinner("Initializing semantic engine…"):
            from sentence_transformers import SentenceTransformer
            st.session_state.embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return st.session_state.embed_model

def get_chroma():
    if "chroma_client" not in st.session_state:
        import chromadb
        st.session_state.chroma_client = chromadb.PersistentClient(path="./dociq_chroma")
    return st.session_state.chroma_client


# ════════════════════════════════════════════════════════════════
# LLM CALL
# ════════════════════════════════════════════════════════════════
def call_llm(messages: list, system: str = "", temperature: float = 0.7, max_tokens: int = 2048) -> str:
    full = []
    if system: full.append({"role": "system", "content": system})
    full.extend(messages)

    if GROQ_API_KEY:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            resp   = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=full, temperature=temperature, max_tokens=max_tokens)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if "rate" not in err and "quota" not in err: return f"[Groq Error] {e}"

    if GEMINI_OK and GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model  = genai.GenerativeModel(model_name="gemini-2.0-flash", system_instruction=system or "You are a helpful assistant.")
            return model.generate_content("\n".join(m["content"] for m in messages)).text.strip()
        except Exception as e: return f"[Gemini Error] {e}"

    return "[Warning] No API key configured."

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.env_log.append(f"[{ts}] {msg}")
    if len(st.session_state.env_log) > 60: st.session_state.env_log.pop(0)


# ════════════════════════════════════════════════════════════════
# PDF PROCESSING
# ════════════════════════════════════════════════════════════════
def ocr_pdf(pdf_bytes: bytes) -> tuple:
    if not OCR_AVAILABLE:
        return "", []
    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
        full_text, page_texts = [], []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang='eng')
            page_texts.append({"page": i+1, "text": text})
            full_text.append(text)
        return "\n".join(full_text).strip(), page_texts
    except Exception as e:
        st.warning(f"OCR failed: {e}")
        return "", []

def parse_pdf(uploaded_file) -> tuple:
    data = uploaded_file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    pages, full = [], []
    for i, page in enumerate(doc):
        txt = page.get_text("text")
        if txt.strip():
            pages.append({"page": i+1, "text": txt})
            full.append(txt)
    doc.close()
    full_text = "\n".join(full).strip()
    num_pages = len(pages)
    if num_pages > 0:
        avg_len = len(full_text) / num_pages
        if avg_len < 50 and OCR_AVAILABLE:
            st.info("Scanned PDF detected — running OCR…")
            ocr_text, ocr_pages = ocr_pdf(data)
            if ocr_text and len(ocr_text) > len(full_text):
                full_text, pages = ocr_text, ocr_pages
        elif avg_len < 50 and not OCR_AVAILABLE:
            st.warning("Scanned PDF detected but OCR is not installed.")
    return full_text, pages

def chunk_text(text: str, size: int = 600, overlap: int = 100) -> list:
    if not text: return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    out, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) <= size:
            cur += " " + s
        else:
            if cur.strip(): out.append(cur.strip())
            cur = (cur[-overlap:] + " " + s) if overlap else s
    if cur.strip(): out.append(cur.strip())
    return out


# ════════════════════════════════════════════════════════════════
# VECTOR STORE
# ════════════════════════════════════════════════════════════════
def col_name(fname: str) -> str:
    h = hashlib.md5(fname.encode()).hexdigest()[:8]
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', fname[:20])
    return f"diq_{safe}_{h}"

def index_pdf(fname: str, chunks: list) -> None:
    if not chunks: return
    chroma_db = get_chroma()
    embed = get_embed_model()
    name = col_name(fname)
    try: chroma_db.delete_collection(name)
    except: pass
    col = chroma_db.create_collection(name, metadata={"hnsw:space": "cosine"})
    embs = embed.encode(chunks, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    ids = [f"{name}_{i}" for i in range(len(chunks))]
    col.add(embeddings=embs.tolist(), documents=chunks, ids=ids)
    gc.collect()

def retrieve(query: str, active_fname: str, top_k: int = 5) -> str | None:
    data = st.session_state.pdf_store.get(active_fname)
    if not data: return None
    chroma_db = get_chroma()
    embed = get_embed_model()
    if not data.get("indexed_for_chat"):
        if not data.get("chunks"): return None
        with st.spinner("Connecting to document memory…"):
            index_pdf(active_fname, data["chunks"])
            st.session_state.pdf_store[active_fname]["indexed_for_chat"] = True
            gc.collect()
    q_emb = embed.encode([query], normalize_embeddings=True)[0].tolist()
    try: col = chroma_db.get_collection(col_name(active_fname))
    except: return None
    n = col.count()
    if n == 0: return None
    res = col.query(query_embeddings=[q_emb], n_results=min(top_k, n))
    hits = [doc for doc, dist in zip(res["documents"][0], res["distances"][0]) if (1.0 - dist) > 0.22]
    return "\n---\n".join(hits) if hits else None


# ════════════════════════════════════════════════════════════════
# DEEP ANALYSIS
# ════════════════════════════════════════════════════════════════
def run_quick_classification(full_text: str) -> dict:
    prompt = (
        'Classify this document. Return ONLY valid JSON, no markdown.\n\n'
        f'Document:\n"""\n{full_text[:3000]}\n"""\n\n'
        'JSON structure:\n'
        '{"doc_type":"Resume|Research Paper|Contract|Report|Article|Transcript|Result|Invoice|Form|Other",'
        '"one_line":"one sentence summary"}'
    )
    raw = call_llm([{"role": "user", "content": prompt}], system="You are a classifier. Return only valid JSON.", temperature=0.1, max_tokens=150)
    try:
        data = json.loads(re.sub(r"```json|```", "", raw).strip())
        data["deep_analysis_done"] = False
        return data
    except:
        return {"doc_type": "Document", "one_line": "Classification complete.", "deep_analysis_done": False}

def run_deep_analysis(full_text: str, existing_analysis: dict) -> dict:
    prompt = (
        'Perform a highly detailed, curated analysis of this document. Extract ONLY vital, critical information.\n\n'
        f'Document:\n"""\n{full_text[:6000]}\n"""\n\n'
        'JSON structure required:\n'
        '{\n'
        '  "doc_type": "Specific Document Type",\n'
        '  "executive_summary": "A dense, high-yield 2-paragraph summary.",\n'
        '  "core_theses": ["Primary thesis/argument 1", "Primary thesis/argument 2"],\n'
        '  "methodologies_and_limitations": ["Method/Limitation 1", "Method/Limitation 2"],\n'
        '  "critical_data_evidence": ["Crucial stat/fact 1", "Crucial metric 2"],\n'
        '  "vital_entities": {\n'
        '    "primary_actors": ["Actor (Why they matter)"],\n'
        '    "key_concepts": ["Concept (Definition)"]\n'
        '  },\n'
        '  "strategic_implications": ["Major implication 1", "Takeaway 2"]\n'
        '}'
    )
    raw = call_llm([{"role": "user", "content": prompt}], system="Expert deep analyst. Return only JSON.", temperature=0.2, max_tokens=2000)
    try:
        new_data = json.loads(re.sub(r"```json|```", "", raw).strip())
        existing_analysis.update(new_data)
        existing_analysis["deep_analysis_done"] = True
        return existing_analysis
    except:
        existing_analysis["executive_summary"] = "Deep analysis failed to parse."
        existing_analysis["deep_analysis_done"] = True
        return existing_analysis


# ════════════════════════════════════════════════════════════════
# DNA FINGERPRINT
# ════════════════════════════════════════════════════════════════
def count_syllables(word: str) -> int:
    word = word.lower()
    count, vowels = 0, "aeiouy"
    if len(word) > 0 and word[0] in vowels: count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i-1] not in vowels: count += 1
    if word.endswith("e"): count -= 1
    return max(count, 1)

def compute_dna(text: str) -> dict:
    if not text or len(text) < 100: return {}
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not sentences or not words: return {}
    wc, sc = max(len(words), 1), max(len(sentences), 1)
    punct_d  = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
    q_d      = text.count('?') / sc
    passive  = len(re.findall(r'\b(was|were|been|is|are)\s+\w+ed\b', text, re.I)) / sc
    conn_d   = sum(text.lower().count(c) for c in ['however','therefore','furthermore','moreover','although','consequently','nevertheless','additionally','thus','hence']) / wc * 100
    fc = sum(text.lower().count(w) for w in ['shall','wherein','hereby','thereof','utilize','implement','leverage','facilitate'])
    cc = sum(text.lower().count(w) for w in ["i'm","you're","can't","won't","don't","it's","gonna","wanna"])
    formality = fc / max(fc + cc + 1, 1)
    syllables = sum(count_syllables(w) for w in words)
    flesch = 206.835 - 1.015 * (wc / sc) - 84.6 * (syllables / wc)
    avg_word = sum(len(w) for w in words) / wc
    def cap(v, s): return round(float(min(v * s, 100.0)), 1)
    return {
        "Reading Ease (Flesch)": max(0, min(flesch, 100)),
        "Lexical Diversity":     round((len(set(words)) / wc) * 100, 1),
        "Word Complexity":       cap(avg_word, 100/8),
        "Punctuation Density":   cap(punct_d, 500),
        "Question Frequency":    cap(q_d, 500),
        "Active Voice Focus":    100 - cap(passive, 300),
        "Transitional Logic":    cap(conn_d, 10),
        "Formality Index":       round(formality * 100, 1),
    }

def get_dna_insights(dna: dict) -> list:
    insights = []
    ease = dna.get("Reading Ease (Flesch)", 0)
    if ease < 30:
        insights.append("Very complex - academic/technical level")
    elif ease < 50:
        insights.append("Moderately complex - college reading level")
    elif ease < 70:
        insights.append("Fairly easy - broad audience suitable")
    else:
        insights.append("Very accessible - conversational style")
    
    formality = dna.get("Formality Index", 0)
    if formality > 70:
        insights.append("Highly formal - legal/academic tone")
    elif formality > 40:
        insights.append("Professional - balanced business tone")
    else:
        insights.append("Informal - conversational style")
    
    active = dna.get("Active Voice Focus", 0)
    if active > 70:
        insights.append("Strong active voice - direct and clear")
    elif active < 40:
        insights.append("Passive voice heavy - objective/scientific style")
    
    diversity = dna.get("Lexical Diversity", 0)
    if diversity > 50:
        insights.append("Rich vocabulary - sophisticated word choice")
    elif diversity < 30:
        insights.append("Limited vocabulary - repetitive language")
    
    return insights

def compare_dna(d1, d2) -> float:
    if not d1 or not d2: return 0.0
    keys = sorted(set(d1) & set(d2))
    v1, v2 = np.array([d1[k] for k in keys], dtype=float), np.array([d2[k] for k in keys], dtype=float)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / n) if n > 0 else 0.0

def dna_radar(dna_data: dict) -> go.Figure:
    cats = ["Reading Ease (Flesch)","Lexical Diversity","Word Complexity","Punctuation Density","Question Frequency","Active Voice Focus","Transitional Logic","Formality Index"]
    colors = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6"]
    fig = go.Figure()
    def hex_rgba(h, a):
        h = h.lstrip('#'); l = len(h)
        return f'rgba({int(h[:l//3],16)},{int(h[l//3:2*l//3],16)},{int(h[2*l//3:],16)},{a})'
    for idx, (fname, dna) in enumerate(dna_data.items()):
        vals = [dna.get(c, 0) for c in cats] + [dna.get(cats[0], 0)]
        c = colors[idx % len(colors)]
        label = (fname[:20]+"…") if len(fname) > 23 else fname
        fig.add_trace(go.Scatterpolar(r=vals, theta=cats+[cats[0]], fill='toself', name=label,
            line=dict(color=c, width=2.5), fillcolor=hex_rgba(c, 0.12)))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0.85)",
            radialaxis=dict(visible=True, range=[0,100], gridcolor="rgba(160,185,230,0.40)", tickfont=dict(color="#6b7a99",size=9)),
            angularaxis=dict(gridcolor="rgba(160,185,230,0.40)", tickfont=dict(color="#0a0f1e",size=10))
        ),
        paper_bgcolor="rgba(244,246,251,0)", plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#0a0f1e", family="Outfit"),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(160,185,230,0.45)", borderwidth=1, font=dict(size=12)),
        margin=dict(l=40,r=40,t=40,b=40), height=400
    )
    return fig


# ════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH
# ════════════════════════════════════════════════════════════════
def extract_triples(full_text: str) -> list:
    if not full_text: return []
    prompt = (
        'Extract 20-25 highly vital entity-relationship triples from this document.\n\n'
        f'Document:\n"""\n{full_text[:4000]}\n"""\n\n'
        'Return ONLY a JSON array:\n'
        '[{"subject":"A","predicate":"RELATION","object":"B","subject_type":"person|org|technology|concept|place","object_type":"person|org|technology|concept|place","importance":1-5}]\n'
    )
    raw = call_llm([{"role": "user", "content": prompt}], system="Knowledge graph builder. Return only JSON.", temperature=0.2, max_tokens=1500)
    try: return json.loads(re.sub(r"```json|```", "", raw).strip())
    except: return []

def analyze_graph(triples: list) -> str:
    if not triples:
        return "No relationships extracted from the document."
    
    entity_counts = {"person": 0, "org": 0, "technology": 0, "concept": 0, "place": 0}
    relationships = []
    subjects = set()
    objects = set()
    
    for t in triples:
        stype = t.get("subject_type", "default")
        otype = t.get("object_type", "default")
        if stype in entity_counts:
            entity_counts[stype] += 1
        if otype in entity_counts:
            entity_counts[otype] += 1
        subjects.add(t.get("subject", ""))
        objects.add(t.get("object", ""))
        relationships.append(t.get("predicate", ""))
    
    analysis = []
    analysis.append(f"**Network Overview:** {len(triples)} relationships connecting {len(subjects)} unique entities.")
    
    active_types = {k: v for k, v in entity_counts.items() if v > 0}
    if active_types:
        top_type = max(active_types, key=active_types.get)
        analysis.append(f"**Dominant Entity Type:** {top_type.title()}s ({active_types[top_type]} occurrences)")
    
    if len(subjects) > 0 and len(objects) > 0:
        avg_connections = len(triples) / max(len(subjects | objects), 1)
        if avg_connections > 3:
            analysis.append(f"**Dense Network:** High interconnectivity ({avg_connections:.1f} connections/entity)")
        else:
            analysis.append(f"**Sparse Network:** Moderate connectivity ({avg_connections:.1f} connections/entity)")
    
    common_rels = Counter(relationships).most_common(3)
    if common_rels:
        rel_summary = ", ".join([f'"{r}" ({c}x)' for r, c in common_rels])
        analysis.append(f"**Frequent Relations:** {rel_summary}")
    
    return "\n\n".join(analysis)

def build_graph_html(triples: list) -> str:
    net = Network(height="550px", width="100%", bgcolor="#f8faff", font_color="#0a0f1e", directed=True)
    net.set_options('{"physics":{"enabled":true,"forceAtlas2Based":{"gravitationalConstant":-80,"centralGravity":0.01,"springLength":120,"springConstant":0.08},"solver":"forceAtlas2Based","stabilization":{"iterations":150}},"edges":{"smooth":{"type":"curvedCW","roundness":0.2},"arrows":{"to":{"enabled":true,"scaleFactor":0.8}}},"interaction":{"hover":true,"zoomView":true,"dragView":true}}')
    type_colors = {"person":"#3b82f6","org":"#10b981","technology":"#8b5cf6","concept":"#f59e0b","place":"#ef4444","default":"#94a3b8"}
    added = set()
    for t in triples:
        s, p, o = t.get("subject","").strip(), t.get("predicate","").strip(), t.get("object","").strip()
        st_, ot_ = t.get("subject_type","default"), t.get("object_type","default")
        importance = t.get("importance", 3)
        if not s or not p or not o: continue
        if s not in added:
            net.add_node(s, label=s, color=type_colors.get(st_,"#94a3b8"), size=20, title=f"{st_}: {s}", font={"size":13})
            added.add(s)
        if o not in added:
            net.add_node(o, label=o, color=type_colors.get(ot_,"#94a3b8"), size=16, title=f"{ot_}: {o}", font={"size":12})
            added.add(o)
        net.add_edge(s, o, label=p, color={"color":"#cbd5e1","highlight":"#3b82f6"}, width=max(1,min(importance,5)), title=f"{p} (Importance: {importance})", font={"size":10,"color":"#6b7a99","align":"middle"})
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
    tmp_name = tmp.name; tmp.close()
    net.save_graph(tmp_name)
    with open(tmp_name, "r", encoding="utf-8") as f: html = f.read()
    os.unlink(tmp_name)
    return html


# ════════════════════════════════════════════════════════════════
# CHAT
# ════════════════════════════════════════════════════════════════
def build_system_prompt(active_fname: str) -> str:
    events = "\n".join(st.session_state.env_log[-5:]) or "Session started."
    return (
        f"You are QueryDoc, an expert document analyst examining: '{active_fname}'.\n"
        "Cite the document precisely when answering. Be analytical, concise, and insightful.\n"
        f"SESSION LOG:\n{events}"
    )

def do_chat(user_msg: str, active_fname: str) -> str:
    ctx = retrieve(user_msg, active_fname)
    model_input = f"[Document Context]\n{ctx}\n\n[Question]: {user_msg}" if ctx else user_msg
    history = st.session_state.pdf_store[active_fname]["chat_history"]
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-20:]]
    msgs.append({"role": "user", "content": model_input})
    reply = call_llm(msgs, system=build_system_prompt(active_fname), temperature=0.7, max_tokens=2048)
    st.session_state.pdf_store[active_fname]["chat_history"].append({"role": "user", "content": user_msg})
    st.session_state.pdf_store[active_fname]["chat_history"].append({"role": "assistant", "content": reply})
    log(f'Chat ({active_fname}): "{user_msg[:40]}"')
    return reply


# ════════════════════════════════════════════════════════════════
# REWRITE
# ════════════════════════════════════════════════════════════════
STYLE_PROMPTS = {
    "Plain Language":   "Rewrite the following text in PLAIN LANGUAGE. Use short sentences (max 15 words), common words, and explain technical terms. Keep all key facts. Make it easy for a beginner to understand.",
    "Interview Format": "Convert the following text into INTERVIEW FORMAT. Create 5-7 question and answer pairs that capture the most important points. Make it conversational, as if someone is being interviewed by an expert.",
    "Professional Post":"Rewrite the following content as a PROFESSIONAL POST suitable for LinkedIn. Start with an attention-grabbing headline, use bullet points for key insights, add 3-5 relevant hashtags at the end. Keep it professional yet engaging. Max 1500 characters.",
    "Editorial Article":"Rewrite the following content as an EDITORIAL ARTICLE. Add an engaging title, use subheadings every few paragraphs, keep paragraphs short (2-3 sentences). Make it readable and shareable for a general audience.",
    "Structured Notes": "Convert the following text into STRUCTURED STUDY NOTES. Use headings, bullet points, definitions, and summaries. Highlight key terms. Organize information hierarchically. Make it easy to review and memorize.",
    "LinkedIn Post":    "Create a VIRAL-READY LINKEDIN POST from the following content. Structure it with: 1) A hook in the first line (question or bold statement), 2) 3-5 bullet points with key insights (use hyphens or asterisks as bullet points), 3) A thoughtful conclusion or call-to-action asking for opinions, 4) 4-6 relevant hashtags at the bottom (#Topic, #Industry, #Insights). Keep the tone professional but conversational. Add line breaks for readability. Target length: 800-1200 characters. Make it engaging and shareable for professionals. Do not use emojis."
}

def rewrite_text(original_text: str, style: str) -> str:
    if not original_text or len(original_text.strip()) < 20:
        return "Error: Text too short to rewrite (minimum 20 characters)."
    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["Plain Language"])
    raw = call_llm(
        [{"role": "user", "content": f"Original text:\n{original_text[:4000]}\n\nRewritten version:"}],
        system=system_prompt, temperature=0.6, max_tokens=2000
    )
    return raw if raw else "Failed to generate rewrite."


# ════════════════════════════════════════════════════════════════
# UI — CHAT TAB
# ════════════════════════════════════════════════════════════════
def render_chat(active_fname: str):
    data = st.session_state.pdf_store[active_fname]
    chat_history = data["chat_history"]

    col_title, col_clear = st.columns([7, 1])
    with col_title:
        short = (active_fname[:52] + "…") if len(active_fname) > 55 else active_fname
        st.markdown(
            f'<div style="font-size:14px;font-weight:600;color:#0a0f1e;padding:2px 0 12px 0;'
            f'letter-spacing:-0.01em;display:flex;align-items:center;gap:8px;">'
            f'<span class="qd-status-dot"></span>{short}</div>',
            unsafe_allow_html=True
        )
    with col_clear:
        if st.button("Clear", help="Clear conversation", use_container_width=True):
            st.session_state.pdf_store[active_fname]["chat_history"] = []
            st.rerun()

    msgs_html = '<div class="chat-scroll-area" id="chatBox" style="padding-bottom: 90px;">'

    if not chat_history:
        msgs_html += (
            '<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;'
            'padding:80px 20px;text-align:center;">'
           
            '<div style="font-size:15px;font-weight:600;color:#0a0f1e;margin-bottom:6px;">Ready to analyse</div>'
            '<div style="font-size:13.5px;color:#6b7a99;line-height:1.7;max-width:340px;">'
            'Ask anything about this document — summaries, facts, comparisons, quotes.</div>'
            '</div>'
        )
    else:
        for msg in chat_history:
            content = msg["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            if msg["role"] == "user":
                msgs_html += f'<div class="chat-wrap-user"><div class="bubble-user">{content}</div></div>'
            else:
                msgs_html += f'<div class="chat-wrap-ai"><div class="avatar-ai">Q</div><div class="bubble-ai">{content}</div></div>'

    msgs_html += '</div>'
    st.markdown(msgs_html, unsafe_allow_html=True)

    st.markdown("""
    <script>
    (function() {
        function scrollBottom() {
            var box = document.getElementById('chatBox');
            if (box) box.scrollTop = box.scrollHeight;
        }
        scrollBottom();
        var obs = new MutationObserver(scrollBottom);
        var target = document.getElementById('chatBox');
        if (target) obs.observe(target, { childList: true, subtree: true });
        window.addEventListener('load', scrollBottom);
    })();
    </script>
    """, unsafe_allow_html=True)

    # Chat input with both Enter and button support
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([10, 1])
        with col1:
            user_input = st.text_input(
                "Message",
                placeholder="Ask anything about this document…",
                label_visibility="collapsed",
                key="chat_input_field"
            )
        with col2:
            submitted = st.form_submit_button("➤", type="primary", use_container_width=True)
        
        if submitted and user_input and user_input.strip():
            msg = user_input.strip()
            if msg != st.session_state.last_sent_msg:
                st.session_state.last_sent_msg = msg
                with st.spinner(""):
                    do_chat(msg, active_fname)
                st.rerun()


# ════════════════════════════════════════════════════════════════
# UI — EXTRACTION TAB (Professional Minimalist)
# ════════════════════════════════════════════════════════════════
def render_analysis(active_fname: str):
    data = st.session_state.pdf_store[active_fname]
    a = data.get("analysis") or {}

    if not a.get("deep_analysis_done", False):
        # Left align button while keeping original width
        col1, col2 = st.columns([10, 10])
        with col1:
            if st.button("Run Structured Extraction", type="primary", use_container_width=False):
                with st.spinner("Extracting insights..."):
                    updated = run_deep_analysis(data["full_text"], a)
                    st.session_state.pdf_store[active_fname]["analysis"] = updated
                st.rerun()
        return

    # Document header
    st.markdown(
        f'<div style="margin-bottom:24px;">'
        f'<div style="font-size:18px;font-weight:600;color:#0a0f1e;margin-bottom:4px;">{active_fname}</div>'
        f'<div style="font-size:11px;color:#6b7a99;text-transform:uppercase;letter-spacing:0.1em;">{a.get("doc_type","Document")}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Executive Summary
    if a.get("executive_summary"):
        st.markdown('<div class="section-label">Executive Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:rgba(255,255,255,0.75);border:1px solid var(--border);border-radius:var(--r-md);padding:20px;margin-bottom:24px;font-size:14px;line-height:1.7;">{a["executive_summary"]}</div>', unsafe_allow_html=True)

    # Two column layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">Core Arguments</div>', unsafe_allow_html=True)
        for arg in a.get("core_theses", []):
            st.markdown(f'<div class="info-item"><div class="info-value">• {arg}</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-label" style="margin-top:24px;">Key Evidence</div>', unsafe_allow_html=True)
        for ev in a.get("critical_data_evidence", []):
            st.markdown(f'<div class="info-item"><div class="info-value">• {ev}</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-label">Methods & Limitations</div>', unsafe_allow_html=True)
        for m in a.get("methodologies_and_limitations", []):
            st.markdown(f'<div class="info-item"><div class="info-value">• {m}</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-label" style="margin-top:24px;">Strategic Implications</div>', unsafe_allow_html=True)
        for imp in a.get("strategic_implications", []):
            st.markdown(f'<div class="info-item"><div class="info-value">• {imp}</div></div>', unsafe_allow_html=True)

    # Key Entities
    vital = a.get("vital_entities", {})
    if vital:
        st.markdown('<div class="section-label">Key Entities</div>', unsafe_allow_html=True)
        ent_col1, ent_col2 = st.columns(2)
        with ent_col1:
            actors = vital.get("primary_actors", [])
            if actors:
                st.markdown('<div class="info-label">Primary Actors</div>', unsafe_allow_html=True)
                for actor in actors[:5]:
                    st.markdown(f'<div class="info-value" style="margin-bottom:8px;">{actor}</div>', unsafe_allow_html=True)
        with ent_col2:
            concepts = vital.get("key_concepts", [])
            if concepts:
                st.markdown('<div class="info-label">Key Concepts</div>', unsafe_allow_html=True)
                for concept in concepts[:5]:
                    st.markdown(f'<div class="info-value" style="margin-bottom:8px;">{concept}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# UI — DNA TAB (Image + Insights side by side)
# ════════════════════════════════════════════════════════════════
def render_dna(active_fname: str):
    dna_store = {f: d["dna"] for f, d in st.session_state.pdf_store.items() if d.get("dna") and not d.get("removed") and d["dna"]}
    if not dna_store:
        st.info("No Writing Style Analysis profile data available. Upload a document first.")
        return
    
    current_dna = dna_store.get(active_fname, {})
    
    if current_dna:
        # Metrics row
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        m_cols = st.columns(4)
        metrics = [
            ("Reading Ease", current_dna.get("Reading Ease (Flesch)", 0)),
            ("Formality", current_dna.get("Formality Index", 0)),
            ("Active Voice", current_dna.get("Active Voice Focus", 0)),
            ("Lexical Diversity", current_dna.get("Lexical Diversity", 0))
        ]
        for i, (label, val) in enumerate(metrics):
            with m_cols[i]:
                st.markdown(f'<div class="metric-box"><div class="metric-val">{val:.0f}</div><div class="metric-lbl">{label}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Radar chart + insights side by side
        chart_col, insight_col = st.columns([1, 1])
        
        with chart_col:
            st.plotly_chart(dna_radar({active_fname: current_dna}), use_container_width=True, key="dna_radar_current")
        
        with insight_col:
            insights = get_dna_insights(current_dna)
            if insights:
                st.markdown('<div class="section-label">Key Insights</div>', unsafe_allow_html=True)
                for insight in insights:
                    st.markdown(f'<div style="background:rgba(255,255,255,0.70);border-left:3px solid #3b82f6;border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:10px;font-size:13px;">{insight}</div>', unsafe_allow_html=True)
    
    # Document comparison
    if len(dna_store) > 1:
        st.markdown('<div class="section-label">Compare Documents</div>', unsafe_allow_html=True)
        default = [active_fname] if active_fname in dna_store else []
        selected = st.multiselect("Select documents to compare", list(dna_store.keys()), default=default)
        if len(selected) >= 2:
            dna_data = {f: dna_store[f] for f in selected}
            st.plotly_chart(dna_radar(dna_data), use_container_width=True, key="dna_radar_compare")
            
            names = list(dna_data.keys())
            matrix = [[round(compare_dna(dna_data[n1], dna_data[n2]) * 100, 1) for n2 in names] for n1 in names]
            shorts = [(n[:20]+"…") if len(n) > 23 else n for n in names]
            fig = px.imshow(matrix, x=shorts, y=shorts,
                color_continuous_scale=[[0,"#e0e7ff"],[0.5,"#93c5fd"],[1,"#2563eb"]],
                zmin=0, zmax=100, text_auto=True)
            fig.update_layout(paper_bgcolor="rgba(244,246,251,0)", plot_bgcolor="rgba(255,255,255,0)", font=dict(color="#0a0f1e", family="Outfit"), height=280, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# UI — GRAPH TAB
# ════════════════════════════════════════════════════════════════
def render_graph(active_fname: str):
    data = st.session_state.pdf_store[active_fname]
    
    # Always show the button prominently
    col_btn, col_spacer = st.columns([2, 3])
    with col_btn:
        if st.button("Build Relationship Map", type="primary", use_container_width=True):
            with st.spinner("Extracting relationships..."):
                triples = extract_triples(data["full_text"])
            with st.spinner("Rendering network..."):
                html = build_graph_html(triples)
                analysis = analyze_graph(triples)
            st.session_state[f"graph_{active_fname}"] = html
            st.session_state[f"triples_{active_fname}"] = triples
            st.session_state[f"graph_analysis_{active_fname}"] = analysis

    html = st.session_state.get(f"graph_{active_fname}")
   
    legend_items = [("#3b82f6","Person"),("#10b981","Organisation"),("#8b5cf6","Technology"),("#f59e0b","Concept"),("#ef4444","Place")]
    legend_html = "".join(f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:18px;font-size:11px;"><span style="width:8px;height:8px;border-radius:50%;background:{c};"></span>{l}</span>' for c, l in legend_items)
    st.markdown(f'<div style="background:rgba(255,255,255,0.78);border:1px solid var(--border);border-radius:12px;padding:10px 16px;margin-bottom:14px;">{legend_html}</div>', unsafe_allow_html=True)
    
    st.components.v1.html(html, height=550, scrolling=False)
    
    analysis = st.session_state.get(f"graph_analysis_{active_fname}")
    if analysis:
        st.markdown('<div class="section-label">Network Analysis</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:rgba(255,255,255,0.75);border:1px solid var(--border);border-radius:var(--r-md);padding:18px;font-size:13px;line-height:1.7;">{analysis}</div>', unsafe_allow_html=True)
        
        triples = st.session_state.get(f"triples_{active_fname}", [])
        if triples:
            with st.expander(f"View all {len(triples)} relationships"):
                for i, t in enumerate(triples[:20], 1):
                    st.markdown(f"**{i}.** {t.get('subject', '')} → *{t.get('predicate', '')}* → {t.get('object', '')}")
                if len(triples) > 20:
                    st.caption(f"... and {len(triples) - 20} more")


# ════════════════════════════════════════════════════════════════
# UI — REWRITE TAB
# ════════════════════════════════════════════════════════════════
def render_rewrite():
    active_pdf = st.session_state.active_pdf
    if not active_pdf:
        st.warning("No document selected.")
        return
    full_text = st.session_state.pdf_store[active_pdf]["full_text"]
    
    source_option = st.radio("Source", ["Entire Document", "Custom Text"], horizontal=True)
    if source_option == "Entire Document":
        source_text = full_text
        st.caption(f"{len(source_text):,} characters — first 4,000 will be processed")
    else:
        source_text = st.text_area("Paste text", height=120, placeholder="Paste text to transform...")
    
    # Align selectbox and button properly
    col_style, col_btn = st.columns([3, 1])
    with col_style:
      selected_style = st.selectbox("Format", list(STYLE_PROMPTS.keys()))
    with col_btn:
    # Add vertical spacing to align button down
      st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
      run = st.button("Rewrite", type="primary", use_container_width=True)

    if run:
        if not source_text.strip():
            st.warning("Please provide text to transform.")
        elif len(source_text.strip()) < 20:
            st.error("Minimum 20 characters required.")
        else:
            with st.spinner(f"Transforming..."):
                result = rewrite_text(source_text, selected_style)
            st.markdown('<div class="section-label">Output</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="rewrite-output">{result}</div>', unsafe_allow_html=True)
            st.download_button(label="Download", data=result, file_name=f"rewrite_{selected_style.replace(' ', '_')}.txt", mime="text/plain")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    with st.sidebar:
        st.markdown('<div style="padding:20px 6px 6px 6px;"><div class="qd-logo">QueryDoc</div><div class="qd-tagline">Your PDF Assistant</div></div>', unsafe_allow_html=True)
        st.divider()

        if not OCR_AVAILABLE:
            st.warning("OCR unavailable. Install pytesseract + pdf2image for scanned PDFs.")

        st.markdown('<span class="upload-label">Upload</span>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

        if uploaded:
            new_uploads = False
            for f in uploaded:
                fname = f.name
                if fname not in st.session_state.pdf_store:
                    full_text, pages = parse_pdf(f)
                    chunks = chunk_text(full_text)
                    dna = compute_dna(full_text)
                    classification = run_quick_classification(full_text) if full_text else {"doc_type":"Document","deep_analysis_done":False}
                    st.session_state.pdf_store[fname] = {
                        "chunks": chunks, "full_text": full_text, "pages": pages,
                        "analysis": classification, "dna": dna, "indexed_for_chat": False,
                        "chat_history": [], "removed": False, "loaded_at": datetime.now().strftime("%H:%M"),
                    }
                    st.session_state.active_pdf = fname
                    new_uploads = True
                    st.toast(f"✓ {fname} loaded")
            if new_uploads:
                st.rerun()

        st.divider()

        active_pdfs = [f for f, d in st.session_state.pdf_store.items() if not d.get("removed")]
        if active_pdfs:
            st.markdown('<span class="doc-list-label">Your Chats</span>', unsafe_allow_html=True)
            if st.session_state.active_pdf not in active_pdfs:
                st.session_state.active_pdf = active_pdfs[0]

            for fname in active_pdfs:
                is_active = (fname == st.session_state.active_pdf)
                c1, c2 = st.columns([5, 1])
                with c1:
                    label = (fname[:28]+"…") if len(fname) > 31 else fname
                    if st.button(label, key=f"nav_{fname}", type="primary" if is_active else "secondary", use_container_width=True):
                        st.session_state.active_pdf = fname
                        st.rerun()
                with c2:
                    if st.button("✕", key=f"del_{fname}"):
                        st.session_state.pdf_store[fname]["removed"] = True
                        if st.session_state.active_pdf == fname:
                            st.session_state.active_pdf = None
                        st.rerun()

        st.markdown('<div style="position:absolute;bottom:18px;left:0;right:0;text-align:center;"></div>', unsafe_allow_html=True)

    if not GROQ_API_KEY:
        st.warning("⚠ No API key found.")

    if not st.session_state.active_pdf:
        st.markdown("""
        <div class="empty-wrap">
            <div class="empty-logo">QueryDoc</div>
            <div style="font-size:16px;font-weight:600;color:#2d3a55;">Your PDF Assistant</div>
            <div class="empty-sub">Upload a PDF from the sidebar to unlock AI-powered conversation, analysis, and more.</div>
            <div><span class="pill">Conversation</span><span class="pill">Extraction</span><span class="pill">Writing Style Analysis</span><span class="pill">Knowledge Graph</span><span class="pill">Rewrite</span></div>
        </div>
        """, unsafe_allow_html=True)
        return

    tabs = st.tabs(["Conversation", "Extraction", "Writing Style Analysis", "Knowledge Graph", "Rewrite"])
    with tabs[0]:
        render_chat(st.session_state.active_pdf)
    with tabs[1]:
        render_analysis(st.session_state.active_pdf)
    with tabs[2]:
        render_dna(st.session_state.active_pdf)
    with tabs[3]:
        render_graph(st.session_state.active_pdf)
    with tabs[4]:
        render_rewrite()


if __name__ == "__main__":
    main()
