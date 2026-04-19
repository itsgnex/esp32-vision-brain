#!/usr/bin/env python3
"""
Local Live Video Reasoning System
=================================
A prototype that captures live video from a Mac webcam, analyzes frames using
a local Ollama vision model, and maintains rolling memory of scene observations.

Requirements:
    pip install opencv-python requests
    brew install ollama
    ollama pull llava

Usage:
    python main.py
"""

import cv2
import json
import time
import os
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip install requests")
    raise


# ============================================================================
# CONFIGURATION SECTION - Edit these values as needed
# ============================================================================

WEBCAM_INDEX = 0  # 0 = default Mac webcam
FRAME_SAMPLE_INTERVAL = 2  # seconds between frame captures
OLLAMA_BASE_URL = "http://localhost:11434"
VISION_MODEL = "llava"  # or "llava:34b", "llava:mistral", etc.
OUTPUT_FOLDER = Path(".")
TERMINAL_LOG = True  # Print updates to terminal

# Prompt template for the vision model
VISION_PROMPT = """Analyze this frame from a live webcam feed. Provide a VERY SHORT structured response in exactly this format:

SUMMARY: [one sentence max]
OBJECTS: [comma separated list, max 5 items, or "none"]
CHANGES: [what changed since last observation, or "none"]

Be brief and focus on what matters."""

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class ObserverState:
    """Maintains rolling memory of scene observations."""
    
    def __init__(self, output_folder: Path):
        self.output_folder = output_folder
        self.current_summary = ""
        self.current_objects = []
        self.last_event = ""
        self.previous_summary = ""
        self.previous_objects = []
        self.history = []
        self.max_history = 10
        self.last_change_detected = ""
        self.event_count = 0
        
        self._load_previous_state()
    
    def _load_previous_state(self):
        """Load previous state from JSON if exists."""
        state_file = self.output_folder / "state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                    self.previous_summary = data.get("latest_summary", "")
                    self.previous_objects = data.get("objects", [])
                    self.history = data.get("history", [])[-self.max_history:]
            except Exception:
                pass
    
    def update(self, new_summary: str, new_objects: list) -> bool:
        """Update state with new observation. Returns True if meaningful change detected."""
        self.previous_summary = self.current_summary
        self.previous_objects = self.current_objects
        self.current_summary = new_summary
        self.current_objects = new_objects
        
        change_detected = False
        change_reason = ""
        
        # Check for meaningful changes
        if new_summary != self.previous_summary:
            # Significant word difference
            if self._compute_similarity(new_summary, self.previous_summary) < 0.7:
                change_detected = True
                change_reason = f"scene changed: {new_summary[:50]}"
        
        # Check if new objects appeared
        new_obj_set = set(new_objects)
        old_obj_set = set(self.previous_objects)
        if new_obj_set != old_obj_set:
            added = new_obj_set - old_obj_set
            removed = old_obj_set - new_obj_set
            if added or removed:
                change_detected = True
                parts = []
                if added:
                    parts.append(f"+{', '.join(list(added)[:3])}")
                if removed:
                    parts.append(f"-{', '.join(list(removed)[:3])}")
                change_reason = f"objects: {' '.join(parts)}"
        
        if change_detected:
            self.last_event = change_reason
            self.last_change_detected = change_reason
            self.event_count += 1
            self.history.append({
                "timestamp": datetime.now().isoformat(),
                "summary": new_summary,
                "change": change_reason
            })
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
        
        return change_detected
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Simple word-overlap similarity (0-1)."""
        if not text1 or not text2:
            return 0.0
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    def write_state_files(self):
        """Write all state files."""
        timestamp = datetime.now().isoformat()
        
        # Write current_state.md
        state_md = self.output_folder / "current_state.md"
        with open(state_md, "w") as f:
            f.write(f"# Current State\n\n")
            f.write(f"**Timestamp:** {timestamp}\n\n")
            f.write(f"**Status:** Active\n\n")
            f.write(f"## Current Scene\n\n")
            f.write(f"{self.current_summary}\n\n")
            f.write(f"## Visible Objects\n\n")
            if self.current_objects:
                f.write(", ".join(self.current_objects))
            else:
                f.write("none detected")
            f.write(f"\n\n## Last Change\n\n")
            f.write(f"{self.last_change_detected or 'none'}\n\n")
            f.write(f"## Event Count: {self.event_count}\n")
        
        # Write event_log.md (append)
        event_log = self.output_folder / "event_log.md"
        is_new = not event_log.exists()
        if self.last_change_detected:
            with open(event_log, "a") as f:
                if is_new:
                    f.write("# Event Log\n\n")
                f.write(f"\n### {timestamp}\n\n")
                f.write(f"{self.last_change_detected}\n\n")
                f.write(f"---\n\n")
        
        # Write state.json
        state_json = self.output_folder / "state.json"
        state_data = {
            "timestamp": timestamp,
            "latest_summary": self.current_summary,
            "objects": self.current_objects,
            "last_event": self.last_change_detected,
            "event_count": self.event_count,
            "history": self.history
        }
        with open(state_json, "w") as f:
            json.dump(state_data, f, indent=2)


# ============================================================================
# VISION ANALYSIS
# ============================================================================

def analyze_frame(frame_path: str, state: ObserverState) -> tuple:
    """Send frame to Ollama and get structured response."""
    import base64
    
    # Encode image as base64
    with open(frame_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
    
    # Build prompt with previous context
    prompt = VISION_PROMPT
    if state.previous_summary:
        prompt = f"Previous observation: {state.previous_summary}\nPrevious objects: {', '.join(state.previous_objects)}\n\n" + prompt
    
    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [img_data],
        "stream": False
    }
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return parse_model_response(result.get("response", ""))
    except requests.exceptions.ConnectionError:
        return ("ERROR: Cannot connect to Ollama. Is Ollama running?", [])
    except requests.exceptions.Timeout:
        return ("ERROR: Ollama request timed out", [])
    except Exception as e:
        return (f"ERROR: {str(e)}", [])


def parse_model_response(text: str) -> tuple:
    """Parse model response into summary and objects."""
    summary = ""
    objects = []
    
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("SUMMARY:"):
            summary = line[8:].strip()
        elif line.startswith("OBJECTS:"):
            obj_str = line[8:].strip()
            if obj_str.lower() != "none" and obj_str:
                objects = [o.strip() for o in obj_str.split(",") if o.strip()]
    
    return (summary, objects)


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    """Main execution loop."""
    print("=" * 60)
    print("Local Live Video Reasoning System")
    print("=" * 60)
    print(f"Webcam index: {WEBCAM_INDEX}")
    print(f"Model: {VISION_MODEL}")
    print(f"Sample interval: {FRAME_SAMPLE_INTERVAL}s")
    print(f"Output folder: {OUTPUT_FOLDER.absolute()}")
    print("=" * 60)
    
    # Ensure output folder exists
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    
    # Initialize webcam
    print("\n[INIT] Opening webcam...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Is it in use by another app?")
        return
    
    # Set reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("[INIT] Webcam opened successfully")
    print("[INIT] Starting continuous capture...\n")
    
    # Initialize state
    state = ObserverState(OUTPUT_FOLDER)
    
    frame_count = 0
    last_sample_time = time.time()
    temp_frame_path = OUTPUT_FOLDER / ".temp_frame.jpg"
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break
            
            frame_count += 1
            
            # Display live preview (optional, shows frame in window)
            cv2.imshow("Live Video Reasoning", frame)
            
            # Check if it's time to sample
            current_time = time.time()
            if current_time - last_sample_time >= FRAME_SAMPLE_INTERVAL:
                last_sample_time = current_time
                
                # Save frame temporarily
                cv2.imwrite(str(temp_frame_path), frame)
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sampling frame #{frame_count}")
                print("[ANALYZ] Sending to Ollama...")
                
                # Analyze frame
                summary, objects = analyze_frame(str(temp_frame_path), state)
                
                print(f"[RESULT] {summary}")
                if objects:
                    print(f"[OBJECTS] {', '.join(objects)}")
                
                # Update state
                change_detected = state.update(summary, objects)
                
                if change_detected:
                    print(f"[CHANGE] {state.last_change_detected}")
                
                # Write state files
                state.write_state_files()
                print("[FILES] State updated")
            
            # Check for quit key (press 'q')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[QUIT] User pressed 'q'")
                break
    
    except KeyboardInterrupt:
        print("\n[QUIT] Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if temp_frame_path.exists():
            temp_frame_path.unlink()
        
        print("\n[EXIT] Cleanup complete")
        print(f"[SUMMARY] Total events logged: {state.event_count}")


if __name__ == "__main__":
    main()