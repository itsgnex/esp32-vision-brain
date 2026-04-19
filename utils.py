"""
Utility functions for the Live Video Reasoning System.
"""

import json
import re
import logging
from typing import Tuple, List, Optional, Dict, Any
from difflib import SequenceMatcher

# Setup logging
logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str] = None, log_to_file: bool = False):
    """Configure logging."""
    level = logging.INFO
    format_str = "%(asctime)s [%(levelname)s] %(message)s"

    if log_to_file and log_file:
        logging.basicConfig(level=level, format=format_str, filename=log_file, filemode="a")
    else:
        logging.basicConfig(level=level, format=format_str)


def log_message(message: str, level: str = "info"):
    """Log message to terminal and optionally to file."""
    getattr(logger, level)(message)


# ============================================================================
# SIMILARITY FUNCTIONS
# ============================================================================

def compute_word_overlap(text1: str, text2: str) -> float:
    """Compute word overlap similarity (0-1)."""
    if not text1 or not text2:
        return 0.0

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def compute_sequence_similarity(text1: str, text2: str) -> float:
    """Compute sequence similarity using SequenceMatcher (0-1)."""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def compute_keyword_similarity(text1: str, text2: str) -> float:
    """Compute similarity based on important keywords only."""
    action_keywords = {"sitting", "standing", "walking", "running", "typing", "looking", "talking",
                  "eating", "drinking", "reading", "writing", "sleeping", "moving", "gesturing"}
    state_keywords = {"empty", "occupied", "active", "idle", "busy", "quiet", "no", "one", "two", "many"}
    object_keywords = {"person", "man", "woman", "child", "desk", "chair", "laptop", "phone",
                     "window", "door", "computer", "book", "cup", "bottle", "food"}

    important_keywords = action_keywords | state_keywords | object_keywords

    words1 = set(text1.lower().split()) & important_keywords
    words2 = set(text2.lower().split()) & important_keywords

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def compute_combined_similarity(text1: str, text2: str) -> float:
    """Compute combined similarity using multiple methods."""
    if not text1 or not text2:
        return 0.0

    word_sim = compute_word_overlap(text1, text2)
    seq_sim = compute_sequence_similarity(text1, text2)
    keyword_sim = compute_keyword_similarity(text1, text2)

    weights = {"word": 0.3, "sequence": 0.3, "keyword": 0.4}
    combined = (
        word_sim * weights["word"] +
        seq_sim * weights["sequence"] +
        keyword_sim * weights["keyword"]
    )

    return combined


def similar_lists(list1: List[str], list2: List[str], threshold: float = 0.7) -> bool:
    """Check if two lists are similar."""
    if not list1 and not list2:
        return True
    if not list1 or not list2:
        return False

    set1 = set(item.lower() for item in list1)
    set2 = set(item.lower() for item in list2)

    if set1 == set2:
        return True

    if len(set1) == 0 and len(set2) == 0:
        return True

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    jaccard = intersection / union if union > 0 else 0.0

    return jaccard >= threshold


# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_json_response(text: str) -> Tuple[Optional[Dict], str]:
    """Parse JSON from model response."""
    text = text.strip()

    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        json_str = json_match.group()
        try:
            data = json.loads(json_str)
            return data, "success"
        except json.JSONDecodeError as e:
            pass

    return None, f"JSON parse error: {str(e)}"


def parse_structured_response(text: str) -> Tuple[Optional[Dict], str]:
    """Parse structured text response as fallback."""
    result = {
        "summary": "",
        "objects": [],
        "actions": [],
        "changes": "none",
        "scene_status": "unknown",
        "confidence": 0.5,
        "important_notes": ""
    }

    current_section = None
    buffer = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        line_lower = line.lower()

        if line_lower.startswith("summary:"):
            result["summary"] = line[8:].strip()
        elif line_lower.startswith("objects:"):
            obj_str = line[8:].strip()
            if obj_str.lower() not in ("none", "[]", ""):
                result["objects"] = [o.strip() for o in obj_str.split(",") if o.strip()]
        elif line_lower.startswith("actions:"):
            action_str = line[8:].strip()
            if action_str.lower() not in ("none", "[]", ""):
                result["actions"] = [a.strip() for a in action_str.split(",") if a.strip()]
        elif line_lower.startswith("changes:"):
            result["changes"] = line[8:].strip()
        elif line_lower.startswith("scene_status:"):
            result["scene_status"] = line[13:].strip()
        elif line_lower.startswith("confidence:"):
            try:
                result["confidence"] = float(line[11:].strip())
            except ValueError:
                pass
        elif line_lower.startswith("important_notes:"):
            result["important_notes"] = line[16:].strip()

    return result if result["summary"] else None, "parsed"


def validate_parsed_response(data: Dict) -> Tuple[bool, str]:
    """Validate that parsed response has required fields."""
    required_fields = ["summary", "objects"]

    for field in required_fields:
        if field not in data:
            return False, f"Missing field: {field}"

    if not data.get("summary"):
        return False, "Empty summary"

    return True, "valid"


def parse_model_response(text: str) -> Tuple[Optional[Dict], str]:
    """Parse model response, trying JSON first, then structured text."""
    data, status = parse_json_response(text)

    if data:
        valid, msg = validate_parsed_response(data)
        if valid:
            return data, "json"

    data, status = parse_structured_response(text)

    if data:
        valid, msg = validate_parsed_response(data)
        if valid:
            return data, "structured"

    return None, "parse_failed"


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_objects(objects: List[str]) -> List[str]:
    """Normalize object list."""
    normalized = []
    for obj in objects:
        obj = obj.strip().lower()
        if obj and obj not in normalized:
            normalized.append(obj)
    return normalized


def normalize_actions(actions: List[str]) -> List[str]:
    """Normalize action list."""
    normalized = []
    for action in actions:
        action = action.strip().lower()
        if action and action not in normalized:
            normalized.append(action)
    return normalized


def normalize_status(status: str) -> str:
    """Normalize scene status."""
    status = status.lower().strip()
    if "active" in status or "movement" in status:
        return "active"
    elif "idle" in status or "empty" in status or "still" in status:
        return "idle"
    else:
        return "unknown"


# ============================================================================
# DIFFERENCE DETECTION
# ============================================================================

def get_object_diff(old: List[str], new: List[str]):
    """Get added and removed objects."""
    old_set = set(o.lower() for o in old)
    new_set = set(o.lower() for o in new)

    added = list(new_set - old_set)
    removed = list(old_set - new_set)

    return added, removed


def get_action_diff(old: List[str], new: List[str]):
    """Get changed actions."""
    old_set = set(a.lower() for a in old)
    new_set = set(a.lower() for a in new)

    added = list(new_set - old_set)
    removed = list(old_set - new_set)

    return added, removed


# ============================================================================
# TIME FORMATTING
# ============================================================================

def format_timestamp(timestamp: str) -> str:
    """Format ISO timestamp for display."""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return timestamp


def get_timestamp() -> str:
    """Get current ISO timestamp."""
    from datetime import datetime
    return datetime.now().isoformat()