#!/usr/bin/env python3
"""
personal_chatbot_backend.py – Azure‑hosted Flask API for the “Friday” chatbot
──────────────────────────────────────────────────────────────────────────────
• RAG over a Qdrant Cloud collection (profile vectors already ingested)
• Local embeddings with Sentence‑Transformers (paraphrase‑multilingual‑mpnet‑base‑v2 → 768‑d)
• Gemini 2.0 Flash for answer generation + back‑translation
• 10‑turn conversation memory per session (RAM‑based)
• Endpoints:
    GET  /             → health‑check
    GET  /ping         → JSON liveness probe
    POST /chat         → {question, session_id?, history?}
    POST /reload       → force re‑ingest PDF into Qdrant
• CORS "*" to simplify frontend calls during testing

Environment variables (set in Azure → Settings → Configuration):
    QDRANT_URL         e.g. https://<cluster‑id>.qdrant.cloud:6333
    QDRANT_API_KEY     ← Qdrant Cloud API key (read/write)
    GOOGLE_API_KEY     ← Gemini key
    QDRANT_COLLECTION  (optional, default="profile")
    PROFILE_DOC_PATH   (optional, default="profile.pdf" in app root)
"""

from __future__ import annotations

# ─── Std‑lib & third‑party imports ─────────────────────────────────────
import os, uuid, logging
from pathlib import Path
from typing import List, Dict
import re
from rag_ingestion import main as reload_ingest

from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import google.generativeai as genai
import pdfplumber
from langdetect import detect_langs
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import MatchText 
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
from functools import lru_cache

# ─── Logging config ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",)
# ensure all DEBUG statements actually print
logging.getLogger().setLevel(logging.DEBUG)

# ─── Environment & constants ───────────────────────────────────────────
load_dotenv()

QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "profile")
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY")
PDF_PATH          = Path(os.getenv("PROFILE_DOC_PATH", "profile.pdf"))

if not (QDRANT_URL and QDRANT_API_KEY and GOOGLE_API_KEY):
    raise RuntimeError("Missing mandatory environment variables (QDRANT_URL / QDRANT_API_KEY / GOOGLE_API_KEY)")

# Embedding model – local in‑process
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMB_DIM    = 768

@lru_cache(maxsize=1)
def get_embed_model() -> SentenceTransformer:
    """Lazily load the embedding model once per process."""
    return SentenceTransformer(MODEL_NAME)

# Retrieval/Generation params
K_RETRIEVE     = 8
SIM_THRESH     = 0.50
MEMORY_LIMIT   = 20
conversation_store: Dict[str, List[Dict[str, str]]] = {}

# ─── External service clients ──────────────────────────────────────────

genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")


def get_qdrant() -> QdrantClient:
    """Light helper to create a fresh Qdrant client when needed."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)



# ─── RAG helpers ───────────────────────────────────────────────────────

def _retrieve_ctx(query: str) -> List[str]:
    client = get_qdrant()
    qvec = get_embed_model().encode(f"query: {query}", convert_to_numpy=True, normalize_embeddings=True).tolist()
    try:
        res = client.search(QDRANT_COLLECTION, query_vector=qvec, limit=K_RETRIEVE)
    except Exception as e:
        logging.error("Qdrant search failed: %s", e)
        return []
    
    logging.debug(f"🔍 Retrieved {len(res)} hits from Qdrant for query: {query!r}")
    for hit in res:
        txt   = hit.payload.get("text", "<no-text>")[:100].replace("\n"," ")
        sect  = hit.payload.get("section", "<no-section>")
        score = 1 - hit.score    # if hit.score is distance; or just hit.score
        logging.debug(f" • id={hit.id} score={score:.3f} section={sect!r}\n   snippet={txt!r}…")

    ctx: List[str] = []
    for p in res:
        sim = 1 - p.score
        if sim >= SIM_THRESH and p.payload and "text" in p.payload:
            ctx.append(p.payload["text"])
    return ctx


def _detect_lang(text: str) -> str:
    try:
        return detect_langs(text)[0].lang
    except Exception:
        return "en"


def _translate(txt: str, tgt: str) -> str:
    if tgt == "en":
        return txt
    return gemini.generate_content(f"Translate to {tgt}:\n\n{txt}").text.strip()


def _build_prompt(
    user_q: str,
    ctx: List[str],
    history: List[Dict[str, str]],
    include_identity: bool = False
    ) -> str:
    ctx_block  = "\n\n---\n\n".join(ctx) if ctx else "No relevant personal data found."
    hist_block = "\n".join(
        f"User: {h['user']}\nFriday: {h['bot']}"
        for h in history[-MEMORY_LIMIT:]
    )
    identity_intro = (
        "Hey! I’m Friday, your friendly AI assistant created by Sai Krishna Prashanth Kolluru.\n"
        "I’m here to help you with any query reagrding Prashanth Kolluru or any general queries.\n\n"
    ) if include_identity else ""
    human_style = (
        "Respond naturally and conversationally and maintain a formal and friendly tone, as if you were a helpful person: "
        "use contractions, vary sentence length, and show empathy when appropriate.\n\n"
    )
    return (
        identity_intro +
        human_style +
        "PERSONAL CONTEXT (if any):\n"
        f"{ctx_block}\n\n"
        "PAST CONVERSATION:\n"
        f"{hist_block}\n\n"
        "USER SAYS:\n"
        f"{user_q}\n"
    )


def _answer(
    translated_q: str,
    sid: str,
    hist_in: List[Dict[str, str]],
    original_q: str = ""
    ) -> str:
    user_q  = original_q or translated_q
    history = conversation_store.setdefault(sid, hist_in)
    ctx     = _retrieve_ctx(translated_q)
    is_first_turn = len(history) == 0
    prompt = _build_prompt(
        user_q,
        ctx,
        history,
        include_identity=is_first_turn
    )
    reply = gemini.generate_content(prompt).text.strip()
    history.append({"user": user_q, "bot": reply})
    if len(history) > MEMORY_LIMIT:
        del history[0]
    return reply


# ─── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors(resp: Response) -> Response:
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "POST,GET,OPTIONS"
    return resp


@app.route("/")
def health() -> str:
    return f"RAG backend ✓ ({MODEL_NAME})", 200


@app.route("/ping")
def ping():
    return jsonify({"status": "alive"}), 200


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    question = (data.get("question")
                or data.get("prompt")
                or data.get("user_prompt")
                or "").strip()
    if not question:
        return jsonify({"error": "empty question"}), 400

    session_id = (data.get("session_id")
                  or request.remote_addr
                  or str(uuid.uuid4())
                  ).strip()
    history = data.get("history") or []

    if len(question) > 2048:
        return jsonify({"error": "input too long"}), 413

    lang = _detect_lang(question)

    # ─── Normalize to English for intent‐matching ────────────────────────
    en_q = question if lang == "en" else gemini.generate_content(
        f"Translate to English:\n\n{question}"
    ).text.strip()

    if not en_q:
        logging.error(f"⚠️ Translation failed for input: {question!r}")
        return jsonify({"error": "Translation failed. Please try again in English."}), 400

    # --- 1) LIST summaries of any major section ------------------------------
    m_list = re.search(
        r'\b(personal projects|projects?|research|education|tools|technologies)\b',
        en_q, re.I
    )
    if m_list:
        kw = m_list.group(1).lower()
        canonical = {
            "personal projects": "personal projects",
            "projects": "personal projects",
            "research": "research journey",
        }.get(kw, kw)

        client = get_qdrant()
        filt = rest.Filter(must=[
            rest.FieldCondition(
                key="section_title",
                match=rest.MatchText(text=canonical)
            ),
            rest.FieldCondition(
                key="is_summary",
                match=rest.MatchValue(value=True)
            )
        ])
        hits = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=get_embed_model()
            .encode(en_q, normalize_embeddings=True)
            .tolist(),
            filter=filt,
            limit=10,
            with_payload=True
        )

        reply_text = "\n".join(
            f"• **{h.payload['section_title']}**: {h.payload['text']}"
            for h in hits
        ) or "I couldn’t find that section in the document."

    # --- 2) DEEP DIVE on any section (e.g., 'tell me more about …') ----------
    elif m_more := re.search(r'(?:tell me (?:more|in detail) about|elaborate on)\s+(.+)', en_q, re.I):
        section_input = m_more.group(1).lower().strip()
        canonical = {
            "personal projects": "personal projects",
            "projects": "personal projects",
            "research": "research journey",
        }.get(section_input, section_input)

        client = get_qdrant()
        filt = rest.Filter(must=[
            rest.FieldCondition(
                key="section_title",
                match=rest.MatchText(text=canonical)
            )
        ])
        hits = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=get_embed_model()
            .encode(en_q, normalize_embeddings=True)
            .tolist(),
            filter=filt,
            limit=10,
            with_payload=True
        )

        reply_text = "\n\n".join(h.payload["text"] for h in hits) or "I couldn't find that section to elaborate on."

    # --- 3) All other queries: fallback to full RAG + Gemini -----------------
    else:
        reply_text = _answer(en_q, sid=session_id, hist_in=history, original_q=question)

    out = reply_text if lang == "en" else f"{reply_text} ({_translate(reply_text, lang)})"
    return jsonify({"answer": out, "language": lang})

@app.route("/reload", methods=["POST"])
def reload_pdf():
    reload_ingest()
    return jsonify({"status": "re-ingested"}), 200

def run_chat(prompt: str, history: List[Dict[str, str]]) -> str:
    return _answer(prompt, sid="azure-function", hist_in=history)

# ─── Local dev runner (ignored in Azure App Service / Function) ───────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)