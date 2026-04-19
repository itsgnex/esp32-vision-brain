"""
Memory module for the Live Video Reasoning System.
Implements layered memory architecture for continuous reasoning.
"""

import json
import logging
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict

# Import config
try:
    import config
except ImportError:
    from . import config

# Import utils
try:
    import utils
except ImportError:
    from . import utils

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """Single observation from the vision system."""
    timestamp: str = ""
    summary: str = ""
    objects: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    changes: str = "none"
    scene_status: str = "unknown"
    confidence: float = 0.5
    important_notes: str = ""
    raw_response: str = ""
    frame_number: int = 0


@dataclass
class Event:
    """Meaningful event for the event log."""
    timestamp: str = ""
    event_type: str = ""
    description: str = ""
    objects_changed: List[str] = field(default_factory=list)
    actions_changed: List[str] = field(default_factory=list)
    scene_status_change: str = ""


class LayeredMemory:
    """Layered memory system for continuous reasoning."""

    def __init__(self, output_folder: Path = None):
        self.output_folder = output_folder or config.OUTPUT_FOLDER
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.current_observation: Optional[Observation] = None
        self.previous_observation: Optional[Observation] = None

        self.recent_context: List[Observation] = []
        self.event_history: List[Event] = []
        self.agent_summary: str = ""

        self.max_recent = config.MAX_RECENT_CONTEXT
        self.max_events = config.MAX_EVENT_HISTORY

        self.min_similarity = config.MIN_SIMILARITY_THRESHOLD
        self.min_action_similarity = config.ACTION_SIMILARITY_THRESHOLD

        self.total_observations = 0
        self.total_events = 0

        self._load_previous_state()

    def _load_previous_state(self) -> None:
        """Load previous state from files."""
        state_file = self.output_folder / "state.json"
        context_file = self.output_folder / "recent_context.json"

        if context_file.exists():
            try:
                with open(context_file) as f:
                    data = json.load(f)
                    context_list = data.get("recent_context", [])
                    self.recent_context = [Observation(**obs) for obs in context_list[-self.max_recent:]]
                    logger.info(f"Loaded {len(self.recent_context)} previous observations")
            except Exception as e:
                logger.warning(f"Failed to load previous context: {e}")

        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                    if data.get("recent_context"):
                        last_ctx = data["recent_context"][-1] if data["recent_context"] else {}
                        if last_ctx:
                            self.agent_summary = last_ctx.get("agent_summary", "")
            except Exception as e:
                logger.warning(f"Failed to load previous state: {e}")

    def add_observation(self, observation: Observation) -> Tuple[bool, Optional[Event]]:
        """Add new observation and detect changes."""
        self.total_observations += 1
        observation.frame_number = self.total_observations

        previous = self.current_observation
        self.previous_observation = previous
        self.current_observation = observation

        change_event = None

        if previous:
            is_meaningful_change, change_type, details = self._detect_change(
                previous, observation
            )

            if is_meaningful_change:
                change_event = Event(
                    timestamp=observation.timestamp,
                    event_type=change_type,
                    description=details.get("description", ""),
                    objects_changed=details.get("objects_added", []) + details.get("objects_removed", []),
                    actions_changed=details.get("actions_changed", []),
                    scene_status_change=details.get("status_change", "")
                )
                self.event_history.append(change_event)
                self.total_events += 1

                if len(self.event_history) > self.max_events:
                    self.event_history = self.event_history[-self.max_events:]

                logger.info(f"Change detected: {change_event.event_type} - {change_event.description}")

        self.recent_context.append(observation)

        if len(self.recent_context) > self.max_recent:
            self.recent_context = self.recent_context[-self.max_recent:]

        return change_event is not None, change_event

    def _detect_change(self, old: Observation, new: Observation) -> Tuple[bool, str, dict]:
        """Detect meaningful changes between observations."""
        details = {}

        summary_changed = not utils.similar_lists(
            [old.summary], [new.summary], self.min_similarity
        )

        objects_added, objects_removed = utils.get_object_diff(
            old.objects, new.objects
        )
        objects_changed = bool(objects_added or objects_removed)

        actions_added, actions_removed = utils.get_action_diff(
            old.actions, new.actions
        )
        actions_changed = bool(actions_added or actions_removed)

        status_changed = old.scene_status != new.scene_status and new.scene_status != "unknown"

        summary_changed = utils.compute_combined_similarity(
            old.summary, new.summary
        ) < self.min_similarity

        if summary_changed:
            return True, "scene_change", {
                "description": f"Scene changed: {new.summary[:100]}",
                "objects_added": objects_added,
                "objects_removed": objects_removed
            }

        if objects_changed:
            parts = []
            if objects_added:
                parts.append(f"+{', '.join(objects_added[:3])}")
            if objects_removed:
                parts.append(f"-{', '.join(objects_removed[:3])}")

            return True, "objects_change", {
                "description": f"Objects: {' '.join(parts)}",
                "objects_added": objects_added,
                "objects_removed": objects_removed
            }

        if actions_changed:
            return True, "actions_change", {
                "description": f"Actions changed: {new.actions}",
                "actions_changed": list(set(actions_added + actions_removed))
            }

        if status_changed:
            return True, "status_change", {
                "description": f"Status: {old.scene_status} -> {new.scene_status}",
                "status_change": f"{old.scene_status} -> {new.scene_status}"
            }

        return False, "none", {}

    def update_agent_summary(self, summary: str) -> None:
        """Update the agent summary."""
        self.agent_summary = summary

    def is_similar_to_recent(self, observation: Observation) -> bool:
        """Check if observation is similar to recent ones (anti-spam)."""
        if not self.recent_context:
            return False

        recent = self.recent_context[-3:]

        for obs in recent:
            if utils.compute_combined_similarity(obs.summary, observation.summary) > 0.85:
                return True

            if utils.similar_lists(obs.objects, observation.objects, 0.85):
                return True

        return False

    def get_current_summary(self) -> str:
        """Get current scene summary."""
        if self.current_observation:
            return self.current_observation.summary
        return ""

    def get_previous_summary(self) -> str:
        """Get previous scene summary."""
        if self.previous_observation:
            return self.previous_observation.summary
        return ""

    def get_current_objects(self) -> List[str]:
        """Get current objects."""
        if self.current_observation:
            return self.current_observation.objects
        return []

    def get_previous_objects(self) -> List[str]:
        """Get previous objects."""
        if self.previous_observation:
            return self.previous_observation.objects
        return []

    def get_current_actions(self) -> List[str]:
        """Get current actions."""
        if self.current_observation:
            return self.current_observation.actions
        return []

    def get_previous_actions(self) -> List[str]:
        """Get previous actions."""
        if self.previous_observation:
            return self.previous_observation.actions
        return []

    def get_recent_context(self, count: int = 5) -> List[Observation]:
        """Get recent context observations."""
        return self.recent_context[-count:]

    def get_event_history(self, count: int = 10) -> List[Event]:
        """Get recent events."""
        return self.event_history[-count:]

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "total_observations": self.total_observations,
            "total_events": self.total_events,
            "recent_context_count": len(self.recent_context),
            "event_history_count": len(self.event_history),
            "current_status": self.current_observation.scene_status if self.current_observation else "none"
        }


def create_observation(data: dict, timestamp: str = None) -> Observation:
    """Create observation from parsed data."""
    if timestamp is None:
        timestamp = utils.get_timestamp()

    return Observation(
        timestamp=timestamp,
        summary=data.get("summary", ""),
        objects=utils.normalize_objects(data.get("objects", [])),
        actions=utils.normalize_actions(data.get("actions", [])),
        changes=data.get("changes", "none"),
        scene_status=utils.normalize_status(data.get("scene_status", "unknown")),
        confidence=data.get("confidence", 0.5),
        important_notes=data.get("important_notes", ""),
        raw_response=data.get("raw_response", "")
    )