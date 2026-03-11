import os
import json
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from app.retriver import load_retriever


load_dotenv(override=True)


class WorkflowState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    # User-facing inputs
    user_query: str
    user_confirmation: Optional[str]
    skip_refinement: bool

    # Query refinement outputs
    refined_query: str
    needs_clarification: bool
    clarifying_questions: List[str]
    assumptions: List[str]
    structured_prompt: str
    confirmed_query: str
    confirmation_status: str

    # Retrieval outputs
    retrieved_context: List[dict]

    # Answer synthesizer outputs
    answer: str
    citations: List[dict]
    reasoning_summary: str

    # Retry loop state
    retry_count: int
    max_retries: int
    retry_target: str
    last_error: str


def _get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _default_max_retries() -> int:
    raw = os.getenv("WORKFLOW_MAX_RETRIES", "2")
    try:
        value = int(raw)
    except ValueError:
        value = 2
    return max(0, value)


def _mark_retry(state: WorkflowState, stage: str, err: Exception) -> WorkflowState:
    count = int(state.get("retry_count", 0)) + 1
    return {
        **state,
        "retry_count": count,
        "retry_target": stage,
        "last_error": str(err),
    }


def _clear_retry(state: WorkflowState) -> WorkflowState:
    return {
        **state,
        "retry_count": 0,
        "retry_target": "",
        "last_error": "",
    }


def _is_obviously_clear_query(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False

    normalized = " ".join(q.lower().split())
    tokens = normalized.split()

    explicit_prefixes = (
        "what is ",
        "who is ",
        "tell me about ",
        "explain ",
        "describe ",
        "overview of ",
        "introduction to ",
    )
    if any(normalized.startswith(prefix) for prefix in explicit_prefixes) and len(tokens) >= 3:
        return True

    ambiguous_markers = {
        "this",
        "that",
        "it",
        "they",
        "those",
        "these",
        "above",
        "below",
        "previous",
        "earlier",
        "same",
    }
    if any(t in ambiguous_markers for t in tokens):
        return False

    if len(tokens) < 8:
        return False

    specific_markers = {"cite", "citations", "page", "pages", "summarize", "compare"}
    return any(t.strip(".,?!") in specific_markers for t in tokens)


def _infer_ambiguous_short_query(query: str) -> Optional[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return None

    clean = " ".join(q.replace("?", " ? ").split())
    words = [w.strip(".,?!").lower() for w in clean.split() if w.strip()]
    if not words:
        return None

    wh_words = {"what", "who", "where", "when", "why", "how"}
    aux_verbs = {"is", "are", "was", "were", "do", "does", "did", "can", "could"}

    if len(words) <= 3 and words[0] in wh_words and (len(words) == 1 or words[1] not in aux_verbs):
        subject = " ".join(w for w in words[1:] if w != "?").strip() or "this topic"
        refined = f"What is {subject}?"
        return {
            "refined_query": refined,
            "structured_prompt": refined,
            "clarifying_questions": [f'Do you mean: "{refined}"?'],
        }

    if len(words) <= 2 and words[0] not in wh_words:
        subject = " ".join(w for w in words if w != "?").strip()
        refined = f"What is {subject}?"
        return {
            "refined_query": refined,
            "structured_prompt": refined,
            "clarifying_questions": [f'Do you mean: "{refined}"?'],
        }

    return None


def query_refinement_agent(state: WorkflowState) -> WorkflowState:
    # Resume path: keep existing refinement results from a prior turn.
    if state.get("skip_refinement") and state.get("structured_prompt"):
        return _clear_retry(state)

    user_query = state.get("user_query", "").strip()
    if not user_query:
        return _mark_retry(state, "query_refinement", ValueError("user_query is required"))

    try:
        llm = _get_chat_model()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a Query Refinement Agent for a document-grounded Q&A system.\n"
                        "Your job is to convert a raw user query into a structured, retrieval-ready form.\n\n"
                        "You MUST respond in valid JSON with the following keys:\n"
                        "  - refined_query: string\n"
                        "  - needs_clarification: boolean\n"
                        "  - clarifying_questions: array of strings (can be empty)\n"
                        "  - assumptions: array of strings (can be empty)\n"
                        "  - structured_prompt: string (a concise, well-formed prompt for retrieval)\n"
                        "\n"
                        "Behaviors:\n"
                        "- Detect ambiguity or missing constraints in the query.\n"
                        "- Set needs_clarification=true ONLY when the question cannot be answered reliably without extra user input.\n"
                        "- If needs_clarification=true, add 1-3 clarifying questions.\n"
                        "- If the query is already specific and answerable, set needs_clarification=false and clarifying_questions=[].\n"
                        "- If you must make reasonable assumptions, list them explicitly in assumptions.\n"
                        "- Do NOT include any explanation outside the JSON object."
                    ),
                ),
                ("user", "Raw user query:\n{query}"),
            ]
        )

        content = (prompt | llm).invoke({"query": user_query}).content.strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Refinement agent returned invalid JSON")

        refined_query = str(parsed.get("refined_query", "")).strip()
        structured_prompt = str(parsed.get("structured_prompt", "")).strip()
        if not refined_query and not structured_prompt:
            raise ValueError("Refinement output missing refined_query/structured_prompt")

        needs_clarification = bool(parsed.get("needs_clarification", False))
        clarifying_questions = parsed.get("clarifying_questions", []) or []

        ambiguous_hint = _infer_ambiguous_short_query(user_query)
        if ambiguous_hint:
            needs_clarification = True
            clarifying_questions = ambiguous_hint["clarifying_questions"]
            refined_query = ambiguous_hint["refined_query"]
            structured_prompt = ambiguous_hint["structured_prompt"]

        if _is_obviously_clear_query(user_query):
            needs_clarification = False
            clarifying_questions = []

        if not needs_clarification:
            clarifying_questions = []

        return {
            **_clear_retry(state),
            "refined_query": refined_query or user_query,
            "needs_clarification": needs_clarification,
            "clarifying_questions": clarifying_questions,
            "assumptions": parsed.get("assumptions", []),
            "structured_prompt": structured_prompt or refined_query or user_query,
        }
    except Exception as e:
        return _mark_retry(state, "query_refinement", e)


def user_confirmation_gate(state: WorkflowState) -> WorkflowState:
    needs_clarification = bool(state.get("needs_clarification", False))
    structured = (
        state.get("structured_prompt")
        or state.get("refined_query")
        or state.get("user_query", "")
    )
    structured = (structured or "").strip()

    if not needs_clarification:
        return {
            **state,
            "confirmed_query": structured,
            "confirmation_status": "confirmed",
        }

    raw_confirmation = (state.get("user_confirmation") or "").strip().lower()
    if raw_confirmation in {"yes", "y"}:
        return {
            **state,
            "confirmed_query": structured,
            "confirmation_status": "confirmed",
        }

    if raw_confirmation in {"no", "n"}:
        return {
            **state,
            "confirmation_status": "rejected",
            "answer": "Sorry, I did not understand the question. Please rephrase it with more detail or specificity.",
            "citations": [],
            "reasoning_summary": "User rejected the refined query; retrieval skipped.",
            "retrieved_context": [],
        }

    return {
        **state,
        "confirmation_status": "awaiting_confirmation",
        "answer": "",
        "citations": [],
        "reasoning_summary": "",
        "retrieved_context": [],
    }


def retrieval_agent(state: WorkflowState) -> WorkflowState:
    try:
        refined_query = state.get("confirmed_query") or state.get(
            "structured_prompt", state.get("refined_query", state.get("user_query", ""))
        )
        refined_query = (refined_query or "").strip()
        if not refined_query:
            raise ValueError("Refined/structured query is required for retrieval")

        retriever = load_retriever()
        vectorstore = retriever.vectorstore
        top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
        results = vectorstore.similarity_search_with_score(refined_query, k=top_k)

        if not results:
            raise ValueError("No retrieval results returned")

        retrieved_context: List[dict] = []
        for idx, (doc, score) in enumerate(results):
            meta = doc.metadata or {}
            retrieved_context.append(
                {
                    "chunk_id": meta.get("doc_id", meta.get("id", f"chunk-{idx}")),
                    "page": meta.get("page"),
                    "source": os.path.basename(meta.get("source", "unknown")),
                    "text": doc.page_content,
                    "score": float(score),
                }
            )

        return {
            **_clear_retry(state),
            "retrieved_context": retrieved_context,
        }
    except Exception as e:
        return _mark_retry(state, "retrieval", e)


def answer_synthesizer_agent(state: WorkflowState) -> WorkflowState:
    try:
        question = state.get("refined_query") or state.get(
            "structured_prompt", state.get("user_query", "")
        )
        question = (question or "").strip()

        retrieved_context = state.get("retrieved_context", [])
        if not question or not retrieved_context:
            raise ValueError("Answer synthesizer requires question and retrieved_context")

        ctx_blocks = []
        for c in retrieved_context:
            label = f"[chunk_id={c.get('chunk_id')}, page={c.get('page')}]"
            ctx_blocks.append(f"{label}\n{c.get('text', '')}")
        context_str = "\n\n---\n\n".join(ctx_blocks)

        llm = _get_chat_model()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an Answer Synthesizer Agent for a document-grounded Q&A system.\n"
                        "You must answer ONLY using the retrieved context below.\n\n"
                        "Return a STRICT JSON object with the following keys:\n"
                        "  - answer: string\n"
                        "  - citations: array of objects with keys {{claim: string, chunk_id: string, page: integer or null}}\n"
                        "  - reasoning_summary: string (brief high-level reasoning, no raw chain-of-thought)\n\n"
                        "Rules:\n"
                        "- Every key claim in the answer should have at least one citation.\n"
                        "- chunk_id and page MUST come from the given context labels.\n"
                        "- Do NOT include any text outside the JSON object.\n"
                    ),
                ),
                (
                    "user",
                    "Question:\n{question}\n\nRetrieved context:\n{context}\n\nRespond ONLY with the JSON object.",
                ),
            ]
        )

        content = (prompt | llm).invoke({"question": question, "context": context_str}).content.strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Answer synthesizer returned invalid JSON")

        answer = str(parsed.get("answer", "")).strip()
        citations = parsed.get("citations", []) or []
        reasoning_summary = str(parsed.get("reasoning_summary", "")).strip()

        if not answer:
            raise ValueError("Answer synthesizer returned empty answer")

        return {
            **_clear_retry(state),
            "answer": answer,
            "citations": citations,
            "reasoning_summary": reasoning_summary,
        }
    except Exception as e:
        return _mark_retry(state, "answer_synthesizer", e)


def retry_router(state: WorkflowState) -> WorkflowState:
    # Pass-through node used for explicit retry-loop routing.
    return state


def retry_fallback(state: WorkflowState) -> WorkflowState:
    error_msg = state.get("last_error", "Unknown workflow error")
    return {
        **state,
        "confirmation_status": "failed_after_retries",
        "answer": "Sorry, I could not process this reliably. Please rephrase and try again.",
        "reasoning_summary": "",
        "citations": [],
        "retrieved_context": [],
        "last_error": error_msg,
    }


def route_after_refinement(state: WorkflowState) -> str:
    if state.get("retry_target") == "query_refinement":
        return "retry_router"
    return "user_confirmation_gate"


def route_after_confirmation(state: WorkflowState) -> str:
    status = state.get("confirmation_status")
    if status == "confirmed":
        return "retrieval"
    return "end"


def route_after_retrieval(state: WorkflowState) -> str:
    if state.get("retry_target") == "retrieval":
        return "retry_router"
    return "answer_synthesizer"


def route_after_synthesis(state: WorkflowState) -> str:
    if state.get("retry_target") == "answer_synthesizer":
        return "retry_router"
    return "end"


def route_from_retry_router(state: WorkflowState) -> str:
    target = (state.get("retry_target") or "").strip()
    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", _default_max_retries()))

    if target and retry_count <= max_retries:
        return target
    return "retry_fallback"


def route_after_retry_fallback(_: WorkflowState) -> str:
    return "end"


def build_orchestrator():
    graph = StateGraph(WorkflowState)

    graph.add_node("query_refinement", query_refinement_agent)
    graph.add_node("user_confirmation_gate", user_confirmation_gate)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("answer_synthesizer", answer_synthesizer_agent)
    graph.add_node("retry_router", retry_router)
    graph.add_node("retry_fallback", retry_fallback)

    graph.set_entry_point("query_refinement")

    graph.add_conditional_edges(
        "query_refinement",
        route_after_refinement,
        {
            "user_confirmation_gate": "user_confirmation_gate",
            "retry_router": "retry_router",
        },
    )

    graph.add_conditional_edges(
        "user_confirmation_gate",
        route_after_confirmation,
        {"retrieval": "retrieval", "end": END},
    )

    graph.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {
            "answer_synthesizer": "answer_synthesizer",
            "retry_router": "retry_router",
        },
    )

    graph.add_conditional_edges(
        "answer_synthesizer",
        route_after_synthesis,
        {"retry_router": "retry_router", "end": END},
    )

    graph.add_conditional_edges(
        "retry_router",
        route_from_retry_router,
        {
            "query_refinement": "query_refinement",
            "retrieval": "retrieval",
            "answer_synthesizer": "answer_synthesizer",
            "retry_fallback": "retry_fallback",
        },
    )

    graph.add_conditional_edges(
        "retry_fallback",
        route_after_retry_fallback,
        {"end": END},
    )

    return graph.compile()


def run_query_refinement(
    user_query: str,
    user_confirmation: Optional[str] = None,
    prior_state: Optional[WorkflowState] = None,
) -> WorkflowState:
    graph = build_orchestrator()
    initial_state: WorkflowState = {
        "retry_count": 0,
        "max_retries": _default_max_retries(),
        "retry_target": "",
        "last_error": "",
    }

    if prior_state:
        initial_state.update(prior_state)
        initial_state["skip_refinement"] = True

    if user_query:
        initial_state["user_query"] = user_query

    if user_confirmation:
        initial_state["user_confirmation"] = user_confirmation

    return graph.invoke(initial_state)


def run_refinement_only(user_query: str) -> WorkflowState:
    graph = StateGraph(WorkflowState)
    graph.add_node("query_refinement", query_refinement_agent)
    graph.set_entry_point("query_refinement")
    graph.set_finish_point("query_refinement")

    app = graph.compile()
    return app.invoke(
        {
            "user_query": user_query,
            "retry_count": 0,
            "max_retries": _default_max_retries(),
            "retry_target": "",
            "last_error": "",
        }
    )
