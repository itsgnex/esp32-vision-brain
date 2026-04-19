#!/usr/bin/env python3
"""
Local Live Video Reasoning System
======================
A robust prototype that captures live video from a Mac webcam, analyzes frames using
a local Ollama vision model, and maintains layered memory of scene observations.

Requirements:
    pip install opencv-python requests
    brew install ollama
    ollama pull llava

Usage:
    python main.py
"""

import sys
import time
import signal
import logging
from pathlib import Path
from datetime import datetime

# Import local modules
try:
    import config
    import camera
    import vision
    import memory
    import state_manager
    import prompts as prompt_module
    import utils
except ImportError:
    print("Error: Required modules not found. Make sure all files are in the same directory.")
    sys.exit(1)

logger = logging.getLogger(__name__)


# ============================================================================
# TERMINAL DISPLAY
# ============================================================================

def print_header():
    """Print startup header."""
    print("=" * 70)
    print("Local Live Video Reasoning System v2.0")
    print("=" * 70)
    print(f"Webcam:           {config.WEBCAM_INDEX}")
    print(f"Model:            {config.VISION_MODEL}")
    print(f"Sample interval:   {config.FRAME_SAMPLE_INTERVAL}s")
    print(f"Output folder:    {config.OUTPUT_FOLDER.absolute()}")
    print("=" * 70)


def print_status(obs, change_detected, event, response_time, frame_count, total_events):
    """Print live status to terminal."""
    timestamp = datetime.now().strftime("%H:%M:%S")

    print(f"\n[{timestamp}] Frame #{frame_count}")
    print(f"[RESULT] {obs.summary[:100]}{'...' if len(obs.summary) > 100 else ''}")

    if obs.objects:
        print(f"[OBJECTS] {', '.join(obs.objects[:5])}")

    if obs.actions:
        print(f"[ACTIONS] {', '.join(obs.actions[:3])}")

    print(f"[STATUS] {obs.scene_status} | Confidence: {obs.confidence:.0%}")

    if change_detected and event:
        print(f"[CHANGE] {event.event_type}: {event.description}")

    print(f"[TIME] Response: {response_time:.2f}s | Events: {total_events}")


def print_error(message, retry_count=0):
    """Print error message."""
    if retry_count > 0:
        print(f"[ERROR] {message} (retry {retry_count})")
    else:
        print(f"[ERROR] {message}")


def print_init(message):
    """Print init message."""
    print(f"[INIT] {message}")


def print_quit():
    """Print quit message."""
    print("\n[QUIT] Shutting down...")


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    """Main execution loop."""
    signal.signal(signal.SIGINT, lambda s, f: print_quit() or sys.exit(0))

    print_header()
    print_init("Starting system...")

    config.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print_init("Initializing camera...")
    cam = camera.Camera(
        index=config.WEBCAM_INDEX,
        additional_indices=config.WEBCAM_ADDITIONAL_INDICES
    )

    if not cam.open():
        print_error("Failed to open webcam. Is it in use by another app?")
        return

    cam.warmup()

    print_init("Initializing vision client...")
    vision_client = vision.VisionClient(
        base_url=config.OLLAMA_BASE_URL,
        model=config.VISION_MODEL
    )

    connected, msg = vision_client.check_connection()
    if not connected:
        print_error(f"Ollama not available: {msg}")
        print_init("Continuing anyway...")

    print_init("Initializing memory system...")
    memory_system = memory.LayeredMemory(config.OUTPUT_FOLDER)

    print_init("Initializing state manager...")
    state_mgr = state_manager.StateManager(config.OUTPUT_FOLDER)

    temp_frame_path = config.OUTPUT_FOLDER / ".temp_frame.jpg"

    frame_count = 0
    last_sample_time = time.time()
    consecutive_errors = 0

    print_init("Starting continuous capture...\n")

    camera.create_display_window()

    try:
        while True:
            ret, frame = cam.read_valid()

            if not ret:
                print_error("Failed to read frame")
                break

            frame_count += 1

            camera.show_frame(frame)

            current_time = time.time()
            if current_time - last_sample_time < config.FRAME_SAMPLE_INTERVAL:
                key = camera.show_frame(frame)
                if key == ord('q'):
                    print_quit()
                    break
                continue

            last_sample_time = current_time

            camera.save_frame(frame, temp_frame_path, quality=85)

            request_start = time.time()

            current_summary = memory_system.get_current_summary()
            current_objects = memory_system.get_current_objects()
            current_actions = memory_system.get_current_actions()

            prompt = prompt_module.build_vision_prompt(
                previous_summary=current_summary,
                previous_objects=current_objects,
                previous_actions=current_actions,
                frame_count=frame_count
            )

            result, status = vision_client.analyze_image(str(temp_frame_path), prompt)

            response_time = time.time() - request_start

            if not result:
                consecutive_errors += 1
                print_error(f"Analysis failed: {status}", consecutive_errors)

                if consecutive_errors >= 3:
                    print_error("Too many errors, pausing for 10s...")
                    time.sleep(10)
                    consecutive_errors = 0

                camera.show_frame(frame)
                continue

            consecutive_errors = 0

            observation = memory.create_observation(result)

            change_detected, event = memory_system.add_observation(observation)

            if change_detected:
                state_mgr.write_event_log(memory_system.event_history)

            if frame_count % 5 == 0:
                state_mgr.write_all(memory_system, event.description if event else "")

            print_status(observation, change_detected, event, response_time, frame_count, memory_system.total_events)

            key = camera.show_frame(frame)
            if key == ord('q'):
                print_quit()
                break

    except KeyboardInterrupt:
        print_quit()
    finally:
        cam.release()
        camera.destroy_windows()

        if temp_frame_path.exists():
            temp_frame_path.unlink()

        state_mgr.write_all(memory_system, "")

        print(f"\n[EXIT] Total observations: {frame_count}")
        print(f"[EXIT] Total events: {memory_system.total_events}")
        print("[EXIT] Cleanup complete")


if __name__ == "__main__":
    main()