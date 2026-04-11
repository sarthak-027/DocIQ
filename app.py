"""
╔══════════════════════════════════════════════════════════════════╗
║                    DocIQ  v2.6                                   ║
║        Advanced Document Intelligence & Deep Analysis            ║
╚══════════════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

# --- TOP OF app.py ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # This runs on Windows/Mac where pysqlite3 isn't needed/installed
    pass

# ════════════════════════════════════════════════════════════════
# API KEYS — HARDCODED
# ════════════════════════════════════════════════════════════════
import streamlit as st
import os
import gc
from pathlib import Path

# Paste your API keys directly here
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# ════════════════════════════════════════════════════════════════

import streamlit as st
import re, json, time, hashlib, tempfile, string
from datetime import datetime
from collections import Counter
import numpy as np

import fitz                              # PyMuPDF
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

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG 
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DocIQ — Document Intelligence",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_branding = """
<style>
/* 1. Hides the standard Streamlit footer */
footer {visibility: hidden;}

/* 2. Hides the 'Hosted with Streamlit' GitHub badge in the bottom right */
.viewerBadge_container {display: none !important;}
[data-testid="stViewerBadge"] {display: none !important;}

/* 3. (Optional) Hides the top right hamburger menu if you want a pure app feel */
/* #MainMenu {visibility: hidden;} */
</style>
"""
st.markdown(hide_streamlit_branding, unsafe_allow_html=True)

st.markdown(hide_footer, unsafe_allow_html=True)
# ════════════════════════════════════════════════════════════════
# CUSTOM CSS - LIGHT THEME & SIDEBAR STYLING
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary-color: #4f46e5;
    --background-color: #f8fafc;
    --secondary-background-color: #ffffff;
    --text-color: #0f172a;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--background-color); color: var(--text-color); }
[data-testid="stSidebar"] { background: var(--secondary-background-color) !important; border-right: 1px solid #e2e8f0; }

/* LOCK SIDEBAR OPEN */
[data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"] { display: none !important; }
#MainMenu, footer { visibility: hidden; }

/* --- SIDEBAR NAVIGATION UI FIXES --- */
[data-testid="stSidebar"] [data-testid="column"] {
    display: flex;
    flex-direction: column;
    justify-content: center;
}
/* Primary (Active) Nav Button */
[data-testid="stSidebar"] button[kind="primary"] {
    background: #4f46e5 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    text-align: left !important;
    padding-left: 16px !important;
    box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2) !important;
}
/* Secondary (Inactive) Nav Button */
[data-testid="stSidebar"] [data-testid="column"]:nth-of-type(1) button[kind="secondary"] {
    background: transparent !important;
    color: #475569 !important;
    border: 1px solid transparent !important;
    text-align: left !important;
    padding-left: 16px !important;
}
[data-testid="stSidebar"] [data-testid="column"]:nth-of-type(1) button[kind="secondary"]:hover {
    background: #f1f5f9 !important;
    color: #0f172a !important;
}
/* Small Delete 'X' Buttons */
[data-testid="stSidebar"] [data-testid="column"]:nth-of-type(2) button {
    background: transparent !important;
    color: #94a3b8 !important;
    border: none !important;
    padding: 0 !important;
    min-height: 40px;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="column"]:nth-of-type(2) button:hover {
    color: #ef4444 !important;
    background: #fee2e2 !important;
    border-radius: 50%;
}


/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #ffffff; border-radius: 12px; padding: 4px; gap: 4px; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; border-radius: 8px; font-weight: 500; font-size: 13px; padding: 8px 16px; }
.stTabs [aria-selected="true"] { background: #f1f5f9 !important; color: #4f46e5 !important; }

/* Cards & Metrics */
.dociq-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin: 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.dociq-card-accent { background: linear-gradient(135deg,#ffffff,#f8fafc); border: 1px solid #c7d2fe; border-radius: 12px; padding: 20px; margin: 8px 0; box-shadow: 0 2px 4px rgba(99,102,241,0.05); }
.metric-box { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.metric-value { font-size: 28px; font-weight: 700; color: #4f46e5; }
.metric-label { font-size: 12px; color: #64748b; margin-top: 4px; }

/* Chat Bubbles */
.chat-user { background: #f1f5f9; border-radius: 12px 12px 4px 12px; padding: 12px 16px; margin: 8px 0; border: 1px solid #e2e8f0; color: #0f172a; }
.chat-ai   { background: #ffffff; border-radius: 12px 12px 12px 4px; padding: 12px 16px; margin: 8px 0; border: 1px solid #e2e8f0; color: #0f172a; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }

/* Buttons & Inputs */
.stButton > button { transition: all 0.2s ease; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea { background: #ffffff !important; border: 1px solid #cbd5e1 !important; border-radius: 8px !important; color: #0f172a !important; }

/* Typography */
.dociq-logo { font-size: 28px; font-weight: 800; background: linear-gradient(135deg,#4f46e5,#3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px; }
.dociq-tagline { font-size: 11px; color: #64748b; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 10px; }
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ════════════════════════════════════════════════════════════════
# DEFERRED HEAVY RESOURCES (Lazy Loading)
# ════════════════════════════════════════════════════════════════
def get_embed_model():
    if "embed_model" not in st.session_state:
        with st.spinner("Initializing AI Brain (First time only)..."):
            from sentence_transformers import SentenceTransformer 
            st.session_state.embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return st.session_state.embed_model

def get_chroma():
    if "chroma_client" not in st.session_state:
        import chromadb 
        st.session_state.chroma_client = chromadb.PersistentClient(path="./dociq_chroma")
    return st.session_state.chroma_client


# ════════════════════════════════════════════════════════════════
# LLM CALL  — Groq primary, Gemini fallback
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
def parse_pdf(uploaded_file) -> tuple:
    data  = uploaded_file.read()
    doc   = fitz.open(stream=data, filetype="pdf")
    pages, full = [], []
    for i, page in enumerate(doc):
        txt = page.get_text("text")
        if txt.strip():
            pages.append({"page": i + 1, "text": txt})
            full.append(txt)
    doc.close()
    return "\n".join(full).strip(), pages

def chunk_text(text: str, size: int = 600, overlap: int = 100) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    out, cur  = [], ""
    for s in sentences:
        if len(cur) + len(s) <= size: cur += " " + s
        else:
            if cur.strip(): out.append(cur.strip())
            cur = (cur[-overlap:] + " " + s) if overlap else s
    if cur.strip(): out.append(cur.strip())
    return out


# ════════════════════════════════════════════════════════════════
# DEFERRED VECTOR STORE 
# ════════════════════════════════════════════════════════════════
def col_name(fname: str) -> str:
    h    = hashlib.md5(fname.encode()).hexdigest()[:8]
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', fname[:20])
    return f"diq_{safe}_{h}"

def index_pdf(fname: str, chunks: list) -> None:
    chroma_db = get_chroma() 
    embed     = get_embed_model() 
    
    name = col_name(fname)
    try: chroma_db.delete_collection(name)
    except Exception: pass
    
    col = chroma_db.create_collection(name, metadata={"hnsw:space": "cosine"})
    embs = embed.encode(chunks, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    ids  = [f"{name}_{i}" for i in range(len(chunks))]
    col.add(embeddings=embs.tolist(), documents=chunks, ids=ids)
    gc.collect()

def retrieve(query: str, active_fname: str, top_k: int = 5) -> str | None:
    data = st.session_state.pdf_store.get(active_fname)
    if not data: 
        return None

    chroma_db = get_chroma()
    embed     = get_embed_model()

    if not data.get("indexed_for_chat"):
        with st.spinner("Initializing AI memory for this document (happens once)..."):
            index_pdf(active_fname, data["chunks"])
            st.session_state.pdf_store[active_fname]["indexed_for_chat"] = True
            gc.collect()

    q_emb = embed.encode([query], normalize_embeddings=True)[0].tolist()
    
    try: 
        col = chroma_db.get_collection(col_name(active_fname))
    except Exception: 
        return None
    
    n = col.count()
    if n == 0: 
        return None
        
    res = col.query(query_embeddings=[q_emb], n_results=min(top_k, n))
    
    hits = []
    for doc, dist in zip(res["documents"][0], res["distances"][0]):
        if (1.0 - dist) > 0.22: 
            hits.append(doc)

    return "\n---\n".join(hits) if hits else None


# ════════════════════════════════════════════════════════════════
# DEEP ANALYSIS MODULE (ENRICHED)
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
    except Exception:
        return {"doc_type": "Document", "one_line": "Classification complete.", "deep_analysis_done": False}


def run_deep_analysis(full_text: str, existing_analysis: dict) -> dict:
    prompt = (
        'Perform a highly detailed, curated analysis of this document. Extract ONLY vital, critical information and actively ignore trivial details.\n\n'
        f'Document:\n"""\n{full_text[:6000]}\n"""\n\n'
        'JSON structure required:\n'
        '{\n'
        '  "doc_type": "Specific Document Type",\n'
        '  "executive_summary": "A dense, high-yield 2-paragraph summary of the core message.",\n'
        '  "core_theses": ["Primary thesis/argument 1", "Primary thesis/argument 2"],\n'
        '  "methodologies_and_limitations": ["Method or Limitation 1", "Method or Limitation 2"],\n'
        '  "critical_data_evidence": ["Crucial stat or fact 1 with context", "Crucial metric 2"],\n'
        '  "vital_entities": {\n'
        '    "primary_actors": ["Actor (Why they matter)"],\n'
        '    "key_concepts": ["Concept (Definition)"]\n'
        '  },\n'
        '  "strategic_implications": ["Major implication or takeaway 1", "Takeaway 2"]\n'
        '}'
    )
    raw = call_llm([{"role": "user", "content": prompt}], system="You are an expert deep document analyst. Return only JSON.", temperature=0.2, max_tokens=2000)
    try:
        new_data = json.loads(re.sub(r"```json|```", "", raw).strip())
        existing_analysis.update(new_data)
        existing_analysis["deep_analysis_done"] = True
        return existing_analysis
    except Exception:
        existing_analysis["executive_summary"] = "Deep analysis failed to parse correctly. Ensure the document contains readable text."
        existing_analysis["deep_analysis_done"] = True
        return existing_analysis


# ════════════════════════════════════════════════════════════════
# DNA FINGERPRINT (ENRICHED METRICS)
# ════════════════════════════════════════════════════════════════
def count_syllables(word: str) -> int:
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if len(word) > 0 and word[0] in vowels: count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels: count += 1
    if word.endswith("e"): count -= 1
    if count == 0: count += 1
    return count

def compute_dna(text: str) -> dict:
    if not text or len(text) < 100: return {}
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
    words     = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not sentences or not words: return {}

    words_count = max(len(words), 1)
    sentences_count = max(len(sentences), 1)
    
    punct_d   = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
    q_d       = text.count('?') / sentences_count
    passive   = len(re.findall(r'\b(was|were|been|is|are)\s+\w+ed\b', text, re.I)) / sentences_count
    conn_d    = sum(text.lower().count(c) for c in ['however','therefore','furthermore','moreover','although','consequently','nevertheless','additionally','thus','hence']) / words_count * 100
    fc        = sum(text.lower().count(w) for w in ['shall','wherein','hereby','thereof','utilize','implement','leverage','facilitate'])
    cc        = sum(text.lower().count(w) for w in ["i'm","you're","can't","won't","don't","it's","gonna","wanna"])
    formality = fc / max(fc + cc + 1, 1)

    syllables = sum(count_syllables(w) for w in words)
    flesch    = 206.835 - 1.015 * (words_count / sentences_count) - 84.6 * (syllables / words_count)
    avg_word  = sum(len(w) for w in words) / words_count

    def cap(v, scale): return round(float(min(v * scale, 100.0)), 1)
    
    return {
        "Reading Ease (Flesch)": max(0, min(flesch, 100)),
        "Lexical Diversity":     round((len(set(words)) / words_count) * 100, 1),
        "Word Complexity":       cap(avg_word, 100/8),
        "Punctuation Density":   cap(punct_d, 500),
        "Question Frequency":    cap(q_d, 500),
        "Active Voice Focus":    100 - cap(passive, 300),
        "Transitional Logic":    cap(conn_d, 10),
        "Formality Index":       round(formality * 100, 1)
    }

def compare_dna(d1: dict, d2: dict) -> float:
    if not d1 or not d2: return 0.0
    keys = sorted(set(d1) & set(d2))
    v1, v2 = np.array([d1[k] for k in keys], dtype=float), np.array([d2[k] for k in keys], dtype=float)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / n) if n > 0 else 0.0

def dna_radar(dna_data: dict) -> go.Figure:
    cats   = ["Reading Ease (Flesch)", "Lexical Diversity", "Word Complexity", "Punctuation Density", "Question Frequency", "Active Voice Focus", "Transitional Logic", "Formality Index"]
    colors = ["#4f46e5","#3b82f6","#10b981","#f59e0b","#ef4444"] 
    fig    = go.Figure()

    def hex_to_rgba(hex_val, opacity):
        hex_val = hex_val.lstrip('#')
        lv = len(hex_val)
        return f'rgba({int(hex_val[0:lv//3],16)}, {int(hex_val[lv//3:2*lv//3],16)}, {int(hex_val[2*lv//3:],16)}, {opacity})'

    for idx, (fname, dna) in enumerate(dna_data.items()):
        vals = [dna.get(c, 0) for c in cats] + [dna.get(cats[0], 0)]
        fig.add_trace(go.Scatterpolar(r=vals, theta=cats + [cats[0]], fill='toself', name=(fname[:20]+"…") if len(fname)>23 else fname, line=dict(color=colors[idx%len(colors)], width=2), fillcolor=hex_to_rgba(colors[idx%len(colors)], 0.15)))

    fig.update_layout(polar=dict(bgcolor="#ffffff", radialaxis=dict(visible=True, range=[0,100], gridcolor="#e2e8f0", tickfont=dict(color="#64748b", size=9)), angularaxis=dict(gridcolor="#e2e8f0", tickfont=dict(color="#0f172a", size=10))), paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff", font=dict(color="#0f172a", family="Inter"), legend=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", borderwidth=1), margin=dict(l=40, r=40, t=40, b=40), height=420)
    return fig


# ════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH (ENRICHED WITH WEIGHTS)
# ════════════════════════════════════════════════════════════════
def extract_triples(full_text: str) -> list:
    prompt = (
        'Extract 25-30 highly vital entity-relationship triples from this document. Ignore trivial connections.\n\n'
        f'Document:\n"""\n{full_text[:4000]}\n"""\n\n'
        'Return ONLY a JSON array:\n'
        '[{"subject":"A","predicate":"RELATION","object":"B","subject_type":"person|org|technology|concept|place","object_type":"person|org|technology|concept|place", "importance": 1-5}]\n'
    )
    raw = call_llm([{"role": "user", "content": prompt}], system="Knowledge graph builder. Return only JSON.", temperature=0.2, max_tokens=1500)
    try: return json.loads(re.sub(r"```json|```", "", raw).strip())
    except Exception: return []

def build_graph_html(triples: list) -> str:
    net = Network(height="550px", width="100%", bgcolor="#ffffff", font_color="#0f172a", directed=True)
    net.set_options('{"physics":{"enabled":true,"forceAtlas2Based":{"gravitationalConstant":-80,"centralGravity":0.01,"springLength":120,"springConstant":0.08},"solver":"forceAtlas2Based","stabilization":{"iterations":150}},"edges":{"smooth":{"type":"curvedCW","roundness":0.2},"arrows":{"to":{"enabled":true,"scaleFactor":0.8}}},"interaction":{"hover":true,"zoomView":true,"dragView":true}}')

    type_colors = {"person":"#8b5cf6","org":"#3b82f6","technology":"#10b981","concept":"#f59e0b","place":"#ef4444","default":"#94a3b8"}
    added = set()

    for t in triples:
        s, p, o = t.get("subject","").strip(), t.get("predicate","").strip(), t.get("object","").strip()
        st_, ot_ = t.get("subject_type","default"), t.get("object_type","default")
        
        importance = t.get("importance", 3)
        edge_width = max(1, min(importance, 5))
        
        if not s or not p or not o: continue
        if s not in added:
            net.add_node(s, label=s, color=type_colors.get(st_, "#94a3b8"), size=20, title=f"{st_}: {s}", font={"size": 13})
            added.add(s)
        if o not in added:
            net.add_node(o, label=o, color=type_colors.get(ot_, "#94a3b8"), size=16, title=f"{ot_}: {o}", font={"size": 12})
            added.add(o)
        
        net.add_edge(s, o, label=p, color={"color": "#6366f1", "highlight": "#4f46e5"}, width=edge_width, title=f"{p} (Importance: {importance})", font={"size": 10, "color": "#475569", "align": "middle"})

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
    tmp_name = tmp.name 
    tmp.close()
    net.save_graph(tmp_name)
    with open(tmp_name, "r", encoding="utf-8") as f: html = f.read()
    os.unlink(tmp_name)
    return html


# ════════════════════════════════════════════════════════════════
# CHAT SYSTEM (PDF-ISOLATED MEMORY)
# ════════════════════════════════════════════════════════════════
def build_system_prompt(active_fname: str) -> str:
    events = "\n".join(st.session_state.env_log[-5:]) or "Session started."
    return (
        f"You are DocIQ, analyzing the document: '{active_fname}'.\n"
        "Cite the document when answering. Be conversational and highly analytical.\n"
        f"SESSION LOG:\n{events}"
    )

def do_chat(user_msg: str, active_fname: str) -> str:
    ctx = retrieve(user_msg, active_fname)
    model_input = f"[PDF Context]\n{ctx}\n\n[Question]: {user_msg}" if ctx else user_msg
    
    history = st.session_state.pdf_store[active_fname]["chat_history"]
    msgs    = [{"role": m["role"], "content": m["content"]} for m in history[-20:]]
    msgs.append({"role": "user", "content": model_input})

    reply = call_llm(msgs, system=build_system_prompt(active_fname), temperature=0.7, max_tokens=2048)
    
    st.session_state.pdf_store[active_fname]["chat_history"].append({"role": "user",      "content": user_msg})
    st.session_state.pdf_store[active_fname]["chat_history"].append({"role": "assistant", "content": reply})
    log(f'Chat ({active_fname}): "{user_msg[:40]}"')
    return reply


# ════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ════════════════════════════════════════════════════════════════
def render_chat(active_fname: str):
    st.markdown(f"### Chat with {active_fname}")
    chat_history = st.session_state.pdf_store[active_fname]["chat_history"]
    
    chat_area = st.container(height=480)
    with chat_area:
        if not chat_history:
            st.markdown(f'<div style="text-align:center;color:#64748b;padding:60px 20px;"><b>DocIQ is ready.</b><br>Ask anything about this document!</div>', unsafe_allow_html=True)
        for msg in chat_history:
            css, who = ("chat-user", "You") if msg["role"] == "user" else ("chat-ai", "DocIQ")
            content = msg["content"].replace("<","&lt;").replace(">","&gt;")
            st.markdown(f'<div class="{css}"><b>{who}</b><br>{content}</div>', unsafe_allow_html=True)
            
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1: user_input = st.text_input("Message", placeholder="Ask anything about this document...", label_visibility="collapsed")
        with col2: submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        msg = user_input.strip()
        if msg != st.session_state.last_sent_msg:
            st.session_state.last_sent_msg = msg
            with st.spinner("Thinking..."): do_chat(msg, active_fname)
        st.rerun()

def render_analysis(active_fname: str):
    st.markdown("### Deep Document Analysis")
    data  = st.session_state.pdf_store[active_fname]
    a     = data.get("analysis") or {}

    if not a.get("deep_analysis_done", False):
        st.info(f"Fast-load completed for {active_fname}. Run a deep analysis to extract comprehensive insights.")
        if st.button("Run Deep Analysis Now", type="primary"):
            with st.spinner("Executing deep extraction (Summary, Theses, Methodologies)..."):
                updated_analysis = run_deep_analysis(data["full_text"], a)
                st.session_state.pdf_store[active_fname]["analysis"] = updated_analysis
            st.rerun()
        return

    st.markdown(
        f'<div class="dociq-card-accent">'
        f'<b style="color:#4f46e5;font-size:18px;">{active_fname}</b><br>'
        f'<span style="color:#64748b;font-weight:600;">{a.get("doc_type","Document")}</span>'
        f'</div>', unsafe_allow_html=True)

    if a.get("executive_summary"):
        st.markdown("#### Executive Summary")
        st.markdown(f'<div class="dociq-card" style="font-size:15px;line-height:1.6;">{a.get("executive_summary","")}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Core Theses & Arguments")
        for arg in a.get("core_theses", []):
            st.markdown(f'<div style="background:#f1f5f9;border-left:3px solid #3b82f6;padding:8px 12px;margin-bottom:8px;border-radius:4px;">{arg}</div>', unsafe_allow_html=True)
            
        st.markdown("#### Critical Data & Evidence")
        for dp in a.get("critical_data_evidence", []):
            st.markdown(f'<div style="background:#ecfdf5;border-left:3px solid #10b981;padding:8px 12px;margin-bottom:8px;border-radius:4px;">{dp}</div>', unsafe_allow_html=True)

        st.markdown("#### Strategic Implications")
        for imp in a.get("strategic_implications", []):
            st.markdown(f'<div style="background:#fdf4ff;border-left:3px solid #d946ef;padding:8px 12px;margin-bottom:8px;border-radius:4px;">{imp}</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("#### Methodologies & Limitations")
        for m in a.get("methodologies_and_limitations", []):
            st.markdown(f'<div style="background:#fffbeb;border-left:3px solid #f59e0b;padding:8px 12px;margin-bottom:8px;border-radius:4px;">{m}</div>', unsafe_allow_html=True)

        st.markdown("#### Vital Entities")
        for etype, vals in (a.get("vital_entities") or {}).items():
            if vals and isinstance(vals, list):
                clean_vals = [str(next(iter(v.values()), "")) if isinstance(v, dict) else str(v) for v in vals]
                clean_vals = [v for v in clean_vals if v]
                if clean_vals: st.markdown(f"**{etype.replace('_', ' ').title()}:** {', '.join(clean_vals)}")

def render_dna(active_fname: str):
    st.markdown("### Document DNA Fingerprinting")
    dna_store = {f: d["dna"] for f, d in st.session_state.pdf_store.items() if d.get("dna") and not d.get("removed")}
    
    selected = st.multiselect("Select documents to compare", list(dna_store.keys()), default=[active_fname])
    if not selected: return
    dna_data = {f: dna_store[f] for f in selected}

    st.plotly_chart(dna_radar(dna_data), use_container_width=True)

    if len(selected) >= 2:
        st.markdown("#### Similarity Matrix")
        names = list(dna_data.keys())
        matrix = [[round(compare_dna(dna_data[n1], dna_data[n2]) * 100, 1) for n2 in names] for n1 in names]
        shorts = [(n[:20]+"...") if len(n) > 23 else n for n in names]
        fig = px.imshow(matrix, x=shorts, y=shorts, color_continuous_scale=[[0,"#f8fafc"],[0.5,"#818cf8"],[1,"#4f46e5"]], zmin=0, zmax=100, text_auto=True)
        fig.update_layout(paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff", font=dict(color="#0f172a", family="Inter"), height=300, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

def render_graph(active_fname: str):
    st.markdown("### Interactive Knowledge Graph")
    data = st.session_state.pdf_store[active_fname]

    if st.button("Generate Graph for Active PDF", type="primary"):
        with st.spinner("Extracting critical weighted relationships..."): triples = extract_triples(data["full_text"])
        with st.spinner("Rendering graph..."): html = build_graph_html(triples)
        st.session_state[f"graph_{active_fname}"] = html
        st.session_state[f"triples_{active_fname}"] = triples

    html = st.session_state.get(f"graph_{active_fname}")
    if not html: return

    legend_items = [("#8b5cf6","Person"),("#3b82f6","Org"),("#10b981","Tech"),("#f59e0b","Concept"),("#ef4444","Place")]
    legend = "".join(f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px;font-size:12px;color:#0f172a;"><span style="width:10px;height:10px;border-radius:50%;background:{c};display:inline-block;"></span>{l}</span>' for c, l in legend_items)
    st.markdown(f'<div style="margin-bottom:8px;background:#ffffff;padding:10px;border-radius:8px;border:1px solid #e2e8f0;">{legend}</div>', unsafe_allow_html=True)
    st.components.v1.html(html, height=580, scrolling=False)


# ════════════════════════════════════════════════════════════════
# MAIN APPLICATION LOOP
# ════════════════════════════════════════════════════════════════
def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown('<div class="dociq-logo">DocIQ</div>', unsafe_allow_html=True)
        st.markdown('<div class="dociq-tagline">Deep Document Analysis</div>', unsafe_allow_html=True)
        st.divider()

        st.markdown("### New Document")
        uploaded = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
        
        if uploaded:
            new_uploads = False
            for f in uploaded:
                fname = f.name
                if fname not in st.session_state.pdf_store:
                    full_text, pages = parse_pdf(f)
                    chunks = chunk_text(full_text)
                    dna = compute_dna(full_text)

                    st.session_state.pdf_store[fname] = {
                        "chunks":    chunks,
                        "full_text": full_text,
                        "pages":     pages,
                        "analysis":  {"doc_type": "Document", "deep_analysis_done": False},
                        "dna":       dna,
                        "indexed_for_chat": False, 
                        "chat_history": [],        
                        "removed":   False,
                        "loaded_at": datetime.now().strftime("%H:%M"),
                    }
                    st.session_state.active_pdf = fname
                    new_uploads = True
                    log(f"PDF loaded instantly: '{fname}'")
                    st.toast(f"{fname} loaded!")
            
            if new_uploads:
                st.rerun()

        st.divider()
        
        active_pdfs = [f for f, d in st.session_state.pdf_store.items() if not d.get("removed")]
        
        if active_pdfs:
            st.markdown("### Your Documents")
            
            if st.session_state.active_pdf not in active_pdfs:
                st.session_state.active_pdf = active_pdfs[0]
                
            for fname in active_pdfs:
                is_active = (fname == st.session_state.active_pdf)
                
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    btn_type = "primary" if is_active else "secondary"
                    if st.button(fname, key=f"nav_{fname}", type=btn_type, use_container_width=True):
                        st.session_state.active_pdf = fname
                        st.rerun()
                
                with col2:
                    if st.button("X", key=f"del_{fname}", help=f"Remove {fname}"):
                        st.session_state.pdf_store[fname]["removed"] = True
                        if st.session_state.active_pdf == fname:
                            st.session_state.active_pdf = None
                        st.rerun()

    # --- MAIN VIEW ---
    if not GROQ_API_KEY:
        st.warning("[Warning] **No API key found.** Please add your Groq API key directly to the GROQ_API_KEY variable at the top of app.py.")

    if not st.session_state.active_pdf:
        st.markdown(
            '<div style="margin-top: 100px; text-align: center;">'
            '<h1 style="font-size:42px;font-weight:800;background:linear-gradient(135deg,#4f46e5,#3b82f6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px;">DocIQ</h1>'
            '<p style="color:#64748b;font-size:16px;letter-spacing:1px;text-transform:uppercase;">Upload a PDF in the sidebar to begin.</p>'
            '</div>', unsafe_allow_html=True)
        return

    tabs = st.tabs(["Chat", "Deep Analysis", "DNA Fingerprint", "Knowledge Graph"])
    
    with tabs[0]: render_chat(st.session_state.active_pdf)
    with tabs[1]: render_analysis(st.session_state.active_pdf)
    with tabs[2]: render_dna(st.session_state.active_pdf)
    with tabs[3]: render_graph(st.session_state.active_pdf)

if __name__ == "__main__":
    main()
