from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import shutil
import os
import json
import hashlib
from typing import Any, Dict, List, Optional

from app.index import create_indexer_from_env
from app.evaluate_ragas import evaluate_with_ragas
from app.langgraph_workflow import run_query_refinement, run_refinement_only
from app.retriver import load_retriever


# ---------------------------
# Simple in-memory session store for confirmation gate
# ---------------------------
SESSIONS: Dict[str, Dict[str, Any]] = {}

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(
    title="Live RAG API",
    description="RAG with live indexing, retrieval, evaluation, and source attribution",
    version="2.2.0",
)

# ---------------------------
# Static + Templates
# ---------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------------------
# Upload directory
# ---------------------------
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
NO_CONTEXT_ANSWER = "Sorry, I could not find relevant information from the given context."
SOURCE_LOCK_PATH = os.getenv("SOURCE_LOCK_PATH", "./data/active_source.json")

# ---------------------------
# Models
# ---------------------------
class AskRequest(BaseModel):
    question: str


class AskRefineRequest(BaseModel):
    question: str


class AskRefineResponse(BaseModel):
    refined_query: str
    clarifying_questions: List[str]
    assumptions: List[str]
    structured_prompt: str


class AskResponse(BaseModel):
    answer: str
    evaluation: Optional[Dict[str, Any]] = None
    sources: List[Dict[str, Any]]
    refined_query: Optional[str] = None
    clarifying_questions: List[str] = []
    assumptions: List[str] = []
    retrieved_context: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    reasoning_summary: Optional[str] = None
    confirmation_status: Optional[str] = None
    retry_count: Optional[int] = 0
    retry_target: Optional[str] = None
    last_error: Optional[str] = None


# ---------------------------
# Health
# ---------------------------
@app.get("/")
def health():
    return {"status": "ready"}


# ---------------------------
# UI
# ---------------------------
@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------
# Refine query only (Query Refinement Agent)
# ---------------------------
@app.post("/refine", response_model=AskRefineResponse)
def refine_question(request: AskRefineRequest):

    state = run_refinement_only(request.question)

    refined_query = state.get("refined_query", request.question)
    clarifying_questions = state.get("clarifying_questions", [])
    assumptions = state.get("assumptions", [])
    structured_prompt = state.get("structured_prompt", refined_query)

    return {
        "refined_query": refined_query,
        "clarifying_questions": clarifying_questions,
        "assumptions": assumptions,
        "structured_prompt": structured_prompt,
    }


# ---------------------------
# Index document
# ---------------------------
@app.post("/index")
def index_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only the provided PDF is allowed as the knowledge source.",
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_hash = _compute_sha256(file_path)
    active_source = _load_active_source()

    if active_source and active_source.get("sha256") != file_hash:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only one assignment PDF can be used as knowledge source. "
                f"Active source: {active_source.get('filename', 'unknown')}"
            ),
        )

    if not active_source:
        _save_active_source(filename=file.filename, sha256=file_hash)

    os.environ["DOCUMENT_PATH"] = file_path
    create_indexer_from_env(reset_store=True)

    return {"message": "Document Uploaded successfully"}


def _compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_active_source() -> Optional[Dict[str, Any]]:
    if not os.path.exists(SOURCE_LOCK_PATH):
        return None
    try:
        with open(SOURCE_LOCK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _save_active_source(filename: str, sha256: str) -> None:
    lock_dir = os.path.dirname(SOURCE_LOCK_PATH) or "."
    os.makedirs(lock_dir, exist_ok=True)
    payload = {"filename": filename, "sha256": sha256}
    with open(SOURCE_LOCK_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _build_ask_response(state: Dict[str, Any], eval_question: str) -> Dict[str, Any]:
    refined_query = state.get("structured_prompt") or state.get(
        "refined_query", eval_question
    )
    clarifying_questions = state.get("clarifying_questions", [])
    assumptions = state.get("assumptions", [])
    retrieved_context = state.get("retrieved_context", [])
    answer = (state.get("answer") or "").strip()
    citations = state.get("citations", [])
    reasoning_summary = state.get("reasoning_summary", "")
    confirmation_status = state.get("confirmation_status")
    retry_count = int(state.get("retry_count", 0) or 0)
    retry_target = state.get("retry_target")
    last_error = state.get("last_error")

    contexts = [c["text"] for c in retrieved_context] if retrieved_context else []

    # Retry-exhausted fallback should not trigger evaluation or source rendering.
    if confirmation_status == "failed_after_retries":
        evaluation = None
        citations = []
        retrieved_context = []
        reasoning_summary = ""
    elif answer:
        try:
            evaluation = evaluate_with_ragas(
                question=eval_question,
                answer=answer,
                contexts=contexts,
            )
        except Exception as e:
            evaluation = {
                "faithfulness_score": 0.0,
                "hallucination": "Unknown",
                "explanation": f"Evaluation failed: {str(e)}",
            }
    else:
        evaluation = None

    # Post-evaluation safety gate: block low-faithfulness / hallucinated answers.
    gate_threshold = float(os.getenv("ANSWER_FAITHFULNESS_THRESHOLD", "0.7"))
    gated_no_context_answer = NO_CONTEXT_ANSWER
    if evaluation and answer:
        faithfulness_raw = evaluation.get("faithfulness_score", 0.0)
        try:
            faithfulness_score = float(faithfulness_raw)
        except (TypeError, ValueError):
            faithfulness_score = 0.0

        hallucination_flag = str(evaluation.get("hallucination", "")).strip().lower()
        is_hallucinated = hallucination_flag in {"yes", "true", "1"}

        if is_hallucinated or faithfulness_score < gate_threshold:
            answer = gated_no_context_answer
            evaluation = None
            reasoning_summary = ""
            citations = []
            retrieved_context = []

    sources: List[Dict[str, Any]] = []
    answer_lower = answer.lower()
    irrelevant_phrases = [
        "i don't know",
        "cannot find",
        "could not find",
        "no relevant",
        "not provided in the context",
        "could not find any relevant information",
    ]

    if answer and not any(phrase in answer_lower for phrase in irrelevant_phrases):
        seen_source_keys = set()
        for c in retrieved_context:
            raw_text = c.get("text") or ""
            snippet = raw_text[:300] + "..."
            source_name = c.get("source", "unknown")
            page_no = c.get("page")
            # Keep retrieval top-k intact, but avoid duplicate source rows in UI.
            dedupe_key = (
                str(source_name).strip().lower(),
                str(page_no),
                " ".join(snippet.split()).lower(),
            )
            if dedupe_key in seen_source_keys:
                continue
            seen_source_keys.add(dedupe_key)
            sources.append(
                {
                    "source": source_name,
                    "page": page_no,
                    "snippet": snippet,
                }
            )
    else:
        # If answer indicates context-miss, do not show citations/context.
        citations = []
        retrieved_context = []

    return {
        "answer": answer,
        "evaluation": evaluation,
        "sources": sources,
        "refined_query": refined_query,
        "clarifying_questions": clarifying_questions,
        "assumptions": assumptions,
        "retrieved_context": retrieved_context,
        "citations": citations,
        "reasoning_summary": reasoning_summary,
        "confirmation_status": confirmation_status,
        "retry_count": retry_count,
        "retry_target": retry_target,
        "last_error": last_error,
    }


def _is_out_of_context_query(query: str) -> bool:
    """
    Lightweight pre-check to avoid refinement/answering for clearly unrelated queries.
    """
    q = (query or "").strip()
    if not q:
        return True

    try:
        retriever = load_retriever()
        results = retriever.vectorstore.similarity_search_with_score(q, k=3)
    except Exception:
        # If probe fails, do not block normal flow.
        return False

    if not results:
        return True

    max_relevance_distance = float(os.getenv("RELEVANCE_MAX_DISTANCE", "1.3"))
    min_score = min(float(score) for _, score in results)

    # Lexical sanity check: some important query token should appear in top contexts.
    combined_text = " ".join((doc.page_content or "").lower() for doc, _ in results)
    tokens = [t.strip(".,?!").lower() for t in q.split()]
    stop = {
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "is",
        "are",
        "the",
        "a",
        "an",
        "about",
        "tell",
        "me",
        "please",
    }
    key_tokens = [t for t in tokens if len(t) >= 3 and t not in stop]
    has_overlap = any(t in combined_text for t in key_tokens) if key_tokens else False

    # Prioritize lexical grounding when we have meaningful query tokens.
    # If key tokens are present in retrieved text, treat as in-context even if vector distance is high.
    if key_tokens:
        return not has_overlap

    # Fallback path for token-poor queries.
    return min_score > max_relevance_distance


# ---------------------------
# Ask question (RAG + Evaluation + Conditional Sources)
# ---------------------------
@app.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequest, request: Request):

    user_query = body.question.strip()
    client_key = request.client.host if request.client else "unknown"
    pending = SESSIONS.get(client_key)

    # Resume pending confirmation flow if user replied yes/no.
    if pending and user_query.lower() in {"yes", "y", "no", "n"}:
        state = run_query_refinement(
            user_query=pending.get("user_query", ""),
            user_confirmation=user_query,
            prior_state=pending,
        )
        del SESSIONS[client_key]

        eval_query = state.get("confirmed_query") or pending.get("user_query", "")
        return _build_ask_response(state, eval_query)

    # Any non yes/no input starts a fresh run and clears stale pending state.
    if pending:
        del SESSIONS[client_key]

    # If query is unrelated to indexed docs, return direct fallback without refinement.
    if _is_out_of_context_query(user_query):
        return {
            "answer": NO_CONTEXT_ANSWER,
            "evaluation": None,
            "sources": [],
            "refined_query": "",
            "clarifying_questions": [],
            "assumptions": [],
            "retrieved_context": [],
            "citations": [],
            "reasoning_summary": "",
            "confirmation_status": "out_of_context",
            "retry_count": 0,
            "retry_target": "",
            "last_error": "",
        }

    state = run_query_refinement(user_query=user_query)
    status = state.get("confirmation_status")

    if status == "awaiting_confirmation":
        SESSIONS[client_key] = state
        return {
            "answer": "",
            "evaluation": None,
            "sources": [],
            "refined_query": state.get("refined_query", user_query),
            "clarifying_questions": state.get("clarifying_questions", []),
            "assumptions": state.get("assumptions", []),
            "retrieved_context": [],
            "citations": [],
            "reasoning_summary": "",
            "confirmation_status": status,
            "retry_count": int(state.get("retry_count", 0) or 0),
            "retry_target": state.get("retry_target", ""),
            "last_error": state.get("last_error", ""),
        }

    return _build_ask_response(state, user_query)
