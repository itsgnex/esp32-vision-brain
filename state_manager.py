"""
State Manager module for the Live Video Reasoning System.
Handles writing all state files with detailed information.
"""

import json
import logging
from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime

# Import config
try:
    import config
except ImportError:
    from . import config

# Import memory
try:
    import memory
except ImportError:
    from . import memory

logger = logging.getLogger(__name__)


class StateManager:
    """Manages all state file outputs."""

    def __init__(self, output_folder: Path = None):
        self.output_folder = output_folder or config.OUTPUT_FOLDER
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def write_current_state(self, obs: Optional[memory.Observation], event_count: int,
                     last_change: str, agent_summary: str = "") -> None:
        """Write detailed current_state.md."""
        filepath = self.output_folder / "current_state.md"
        timestamp = datetime.now().isoformat()

        status_val = "Active" if obs else "Idle"

        summary_text = obs.summary if obs and obs.summary else "No observation yet"
        objects_text = ", ".join(obs.objects) if obs and obs.objects else "none detected"
        actions_text = ", ".join(obs.actions) if obs and obs.actions else "none detected"
        status_scene = obs.scene_status if obs else "unknown"
        confidence_text = f"{obs.confidence:.2f}" if obs else "N/A"
        notes_text = obs.important_notes if obs and obs.important_notes else "none"
        change_text = last_change if last_change else "none"
        recent_summary = self._format_recent_summary(obs, 3)
        agent_text = agent_summary if agent_summary else "No agent summary yet"
        interpretation = self._generate_interpretation(obs)

        content = f"""# Current State

**Timestamp:** {timestamp}
**Status:** {status_val}

## Current Scene

{summary_text}

## Visible Objects

{objects_text}

## Detected Actions

{actions_text}

## Scene Status: {status_scene}

## Confidence: {confidence_text}

## Important Notes

{notes_text}

## Last Change

{change_text}

## Event Count: {event_count}

## Recent Context Window

### Last 3 Observations

{recent_summary}

---

## Agent Summary

{agent_text}

## Current Interpretation

{interpretation}
"""

        try:
            with open(filepath, "w") as f:
                f.write(content)
            logger.debug(f"Updated {filepath}")
        except Exception as e:
            logger.error(f"Failed to write current_state.md: {e}")

    def _format_recent_summary(self, obs: Optional[memory.Observation], count: int) -> str:
        """Format recent observations for display."""
        if not obs:
            return "No observation yet"

        lines = []
        lines.append(f"- **Objects:** {', '.join(obs.objects) if obs.objects else 'none'}")
        lines.append(f"- **Actions:** {', '.join(obs.actions) if obs.actions else 'none'}")
        lines.append(f"- **Status:** {obs.scene_status}")

        return "\n".join(lines)

    def _generate_interpretation(self, obs: Optional[memory.Observation]) -> str:
        """Generate human-readable interpretation of current state."""
        if not obs or not obs.summary:
            return "No scene data available for interpretation."

        parts = []

        if obs.scene_status == "active":
            parts.append("Scene is active with movement or people.")
        elif obs.scene_status == "idle":
            parts.append("Scene is idle or unoccupied.")
        else:
            parts.append("Scene status is uncertain.")

        if obs.objects:
            if any(p in obs.objects for p in ["person", "man", "woman", "child"]):
                parts.append("People are visible in the scene.")
            if "laptop" in obs.objects or "computer" in obs.objects:
                parts.append("Computing activity likely.")
            if "phone" in obs.objects or "mobile" in obs.objects:
                parts.append("Mobile device detected.")

        if obs.actions:
            parts.append(f"Current activity: {', '.join(obs.actions[:2])}.")

        if obs.confidence < 0.5:
            parts.append(f"Note: Low confidence ({obs.confidence:.0%}) in analysis.")

        return " ".join(parts)

    def write_event_log(self, events: List[memory.Event]) -> None:
        """Write append-only event_log.md."""
        filepath = self.output_folder / "event_log.md"
        is_new = not filepath.exists()

        if not events:
            return

        try:
            with open(filepath, "a") as f:
                if is_new:
                    f.write("# Event Log\n\n")

                for event in events[-1:]:
                    f.write(f"### {event.timestamp}\n\n")
                    f.write(f"**{event.event_type}**: {event.description}\n\n")

                    if event.objects_changed:
                        f.write(f"Objects: {', '.join(event.objects_changed)}\n\n")
                    if event.actions_changed:
                        f.write(f"Actions: {', '.join(event.actions_changed)}\n\n")

                    f.write("---\n\n")

            logger.debug(f"Updated event_log.md ({len(events)} events)")
        except Exception as e:
            logger.error(f"Failed to write event_log.md: {e}")

    def write_state_json(self, memory_system: memory.LayeredMemory) -> None:
        """Write comprehensive state.json."""
        filepath = self.output_folder / "state.json"
        timestamp = datetime.now().isoformat()

        obs = memory_system.current_observation

        data = {
            "timestamp": timestamp,
            "current": {
                "summary": obs.summary if obs else "",
                "objects": obs.objects if obs else [],
                "actions": obs.actions if obs else [],
                "changes": obs.changes if obs else "none",
                "scene_status": obs.scene_status if obs else "unknown",
                "confidence": obs.confidence if obs else 0.0,
                "important_notes": obs.important_notes if obs else ""
            },
            "previous": {
                "summary": memory_system.get_previous_summary(),
                "objects": memory_system.get_previous_objects(),
                "actions": memory_system.get_previous_actions()
            },
            "last_event": memory_system.event_history[-1].event_type if memory_system.event_history else "none",
            "event_count": memory_system.total_events,
            "recent_context": [
                {
                    "timestamp": obs.timestamp,
                    "summary": obs.summary,
                    "objects": obs.objects,
                    "actions": obs.actions,
                    "scene_status": obs.scene_status,
                    "confidence": obs.confidence
                }
                for obs in memory_system.get_recent_context()
            ],
            "agent_summary": memory_system.agent_summary,
            "stats": memory_system.get_stats()
        }

        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Updated state.json")
        except Exception as e:
            logger.error(f"Failed to write state.json: {e}")

    def write_recent_context_md(self, memory_system: memory.LayeredMemory) -> None:
        """Write recent_context.md for human readability."""
        filepath = self.output_folder / "recent_context.md"
        timestamp = datetime.now().isoformat()

        recent = memory_system.get_recent_context(10)

        if not recent:
            return

        content = f"# Recent Context\n\n**Last Updated:** {timestamp}\n\n"

        for i, obs in enumerate(recent):
            content += f"## Observation {i + 1} [{obs.timestamp}]\n\n"
            content += f"**Summary:** {obs.summary}\n\n"
            content += f"**Objects:** {', '.join(obs.objects) if obs.objects else 'none'}\n\n"
            content += f"**Actions:** {', '.join(obs.actions) if obs.actions else 'none'}\n\n"
            content += f"**Status:** {obs.scene_status}\n\n"
            content += f"**Confidence:** {obs.confidence:.2f}\n\n"

            if obs.important_notes:
                content += f"**Notes:** {obs.important_notes}\n\n"

            content += "---\n\n"

        try:
            with open(filepath, "w") as f:
                f.write(content)
            logger.debug(f"Updated recent_context.md")
        except Exception as e:
            logger.error(f"Failed to write recent_context.md: {e}")

    def write_recent_context_json(self, memory_system: memory.LayeredMemory) -> None:
        """Write recent_context.json for machine use."""
        filepath = self.output_folder / "recent_context.json"
        timestamp = datetime.now().isoformat()

        recent = memory_system.get_recent_context()

        data = {
            "timestamp": timestamp,
            "recent_context": [
                {
                    "timestamp": obs.timestamp,
                    "summary": obs.summary,
                    "objects": obs.objects,
                    "actions": obs.actions,
                    "changes": obs.changes,
                    "scene_status": obs.scene_status,
                    "confidence": obs.confidence,
                    "important_notes": obs.important_notes
                }
                for obs in recent
            ],
            "agent_summary": memory_system.agent_summary
        }

        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Updated recent_context.json")
        except Exception as e:
            logger.error(f"Failed to write recent_context.json: {e}")

    def write_agent_summary(self, summary: str) -> None:
        """Write agent_summary.md."""
        filepath = self.output_folder / "agent_summary.md"
        timestamp = datetime.now().isoformat()

        content = f"# Agent Summary\n\n**Last Updated:** {timestamp}\n\n{summary}\n"

        try:
            with open(filepath, "w") as f:
                f.write(content)
            logger.debug(f"Updated agent_summary.md")
        except Exception as e:
            logger.error(f"Failed to write agent_summary.md: {e}")

    def write_all(self, memory_system: memory.LayeredMemory, last_change: str = "") -> None:
        """Write all state files."""
        obs = memory_system.current_observation
        event_count = memory_system.total_events
        agent_summary = memory_system.agent_summary

        self.write_current_state(obs, event_count, last_change, agent_summary)
        self.write_event_log(memory_system.event_history)
        self.write_state_json(memory_system)
        self.write_recent_context_md(memory_system)
        self.write_recent_context_json(memory_system)

        if agent_summary:
            self.write_agent_summary(agent_summary)