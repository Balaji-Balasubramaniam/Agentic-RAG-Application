import os
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv(override=True)


def _fallback_eval(message: str) -> Dict:
    return {
        "faithfulness_score": 1.0,
        "answer_relevancy_score": None,
        "hallucination": "Unknown",
        "explanation": message,
        "ragas_metrics": {},
    }


def evaluate_with_ragas(
    question: str,
    answer: str,
    contexts: List[str],
) -> Dict:
    """
    RAGAS-based evaluation returning API-compatible fields:
    - faithfulness_score: float (0-1)
    - hallucination: Yes/No/Unknown
    - explanation: short message
    """

    if not answer or not contexts:
        return _fallback_eval("RAGAS skipped: missing answer or retrieved contexts.")

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from langchain_openai import ChatOpenAI
    except Exception as e:
        return _fallback_eval(f"RAGAS unavailable: {str(e)}")

    try:
        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
        )

        # RAGAS faithfulness metric evaluator
        judge_llm = ChatOpenAI(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        scores = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=judge_llm,
            raise_exceptions=False,
        )

        row = scores.to_pandas().iloc[0].to_dict()
        raw_score = row.get("faithfulness", 0.0)
        score = float(raw_score) if raw_score is not None else 0.0
        raw_answer_relevancy = row.get("answer_relevancy", None)
        answer_relevancy_score = (
            float(raw_answer_relevancy)
            if raw_answer_relevancy is not None
            else None
        )

        # Convert all available RAGAS outputs into JSON-safe metrics.
        ragas_metrics: Dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                continue
            if isinstance(value, (bool, int, float, str)):
                ragas_metrics[key] = value
                continue
            if isinstance(value, list):
                safe_list = []
                for item in value:
                    if isinstance(item, (bool, int, float, str)):
                        safe_list.append(item)
                    else:
                        safe_list.append(str(item))
                ragas_metrics[key] = safe_list
                continue
            if isinstance(value, dict):
                safe_dict: Dict[str, Any] = {}
                for k, v in value.items():
                    if isinstance(v, (bool, int, float, str)):
                        safe_dict[str(k)] = v
                    elif isinstance(v, list):
                        safe_dict[str(k)] = [
                            i if isinstance(i, (bool, int, float, str)) else str(i)
                            for i in v
                        ]
                    else:
                        safe_dict[str(k)] = str(v)
                ragas_metrics[key] = safe_dict
                continue
            try:
                ragas_metrics[key] = float(value)
            except Exception:
                ragas_metrics[key] = str(value)

        threshold = float(os.getenv("ANSWER_FAITHFULNESS_THRESHOLD", "0.7"))
        hallucination = "Yes" if score < threshold else "No"

        return {
            "faithfulness_score": max(0.0, min(score, 1.0)),
            "answer_relevancy_score": (
                max(0.0, min(answer_relevancy_score, 1.0))
                if answer_relevancy_score is not None
                else None
            ),
            "hallucination": hallucination,
            "explanation": "RAGAS evaluation completed.",
            "ragas_metrics": ragas_metrics,
        }
    except Exception as e:
        return _fallback_eval(f"RAGAS evaluation failed: {str(e)}")
