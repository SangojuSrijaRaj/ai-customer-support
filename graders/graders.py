from typing import Dict, Any, Tuple


def _score_urgency(predicted: str, ground_truth: str) -> float:
    levels = ["low", "medium", "high", "critical"]
    if predicted == ground_truth:
        return 1.0
    try:
        dist = abs(levels.index(predicted) - levels.index(ground_truth))
        return max(0.0, 1.0 - dist * 0.4)
    except ValueError:
        return 0.0


def _score_department(predicted: str, ground_truth: str) -> float:
    return 1.0 if predicted == ground_truth else 0.0


def _score_reply(reply: str, keywords: list) -> float:
    if not reply or not keywords:
        return 0.0
    reply_lower = reply.lower()
    matched = sum(1 for kw in keywords if kw.lower() in reply_lower)
    return round(matched / len(keywords), 2)


def _score_escalation(predicted: bool, ground_truth: bool) -> float:
    return 1.0 if predicted == ground_truth else 0.0


def grade_easy(action: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict]:
    u = _score_urgency(action.get("urgency", ""), ground_truth["urgency"])
    d = _score_department(action.get("department", ""), ground_truth["department"])
    total = round(0.5 * u + 0.5 * d, 3)
    return total, {"urgency_score": u, "department_score": d,
                   "predicted_urgency": action.get("urgency"), "expected_urgency": ground_truth["urgency"],
                   "predicted_department": action.get("department"), "expected_department": ground_truth["department"]}


def grade_medium(action: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict]:
    u = _score_urgency(action.get("urgency", ""), ground_truth["urgency"])
    d = _score_department(action.get("department", ""), ground_truth["department"])
    r = _score_reply(action.get("reply_draft", ""), ground_truth.get("reply_keywords", []))
    total = round(0.3 * u + 0.3 * d + 0.4 * r, 3)
    return total, {"urgency_score": u, "department_score": d, "reply_score": r}


def grade_hard(action: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict]:
    u = _score_urgency(action.get("urgency", ""), ground_truth["urgency"])
    d = _score_department(action.get("department", ""), ground_truth["department"])
    r = _score_reply(action.get("reply_draft", ""), ground_truth.get("reply_keywords", []))
    e = _score_escalation(action.get("escalate", False), ground_truth.get("escalate", False))
    total = round(0.2 * u + 0.2 * d + 0.3 * r + 0.3 * e, 3)
    return total, {"urgency_score": u, "department_score": d, "reply_score": r, "escalation_score": e}


TASK_GRADERS = {
    "task_easy": grade_easy,
    "task_medium": grade_medium,
    "task_hard": grade_hard,
}
