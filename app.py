#!/usr/bin/env python3
"""
Live Video Reasoning Web App
============================
Flask-based web interface for real-time video analysis with Ollama.
"""

import os
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

import cv2
import time
import threading
import base64
from datetime import datetime
from flask import Flask, render_template, jsonify, request

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip install requests")
    raise

app = Flask(__name__)

WEBCAM_INDEX = 0
OLLAMA_BASE_URL = "http://localhost:11434"
VISION_MODEL = "bakllava:latest"
VISION_PROMPT = """Analyze this frame from a live webcam feed. Provide a VERY SHORT structured response in exactly this format:

SUMMARY: [one sentence max]
OBJECTS: [comma separated list, max 5 items, or "none"]
CHANGES: [what changed since last observation, or "none"]

Be brief and focus on what matters."""

context_log = []
max_context_length = 50
lock = threading.Lock()
current_frame = None
current_frame_lock = threading.Lock()

capture_interval_ms = 2000
is_capturing = True
is_analyzing = False
analysis_thread = None


class ContextEntry:
    def __init__(self, image_data, summary, objects, timestamp):
        self.image_data = image_data
        self.summary = summary
        self.objects = objects
        self.timestamp = timestamp


def capture_frame():
    global current_frame, is_capturing
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    print("[INIT] Webcam opened successfully")

    while is_capturing:
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            with current_frame_lock:
                current_frame = buffer.tobytes()

        time.sleep(0.03)

    cap.release()
    print("[CAPTURE] Stopped frame capture")


def analyze_frame(image_data):
    img_data = base64.b64encode(image_data).decode("utf-8")

    prompt = VISION_PROMPT
    with lock:
        if context_log:
            last_entry = context_log[-1]
            prompt = f"Previous observation: {last_entry.summary}\nPrevious objects: {', '.join(last_entry.objects)}\n\n" + VISION_PROMPT

    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [img_data],
        "stream": False
    }

    print(f"[ANALYSIS] Sending to {VISION_MODEL}...")

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120
        )
        print(f"[ANALYSIS] Response status: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        raw_response = result.get("response", "")
        print(f"[ANALYSIS] Raw: {raw_response[:100]}...")
        return parse_model_response(raw_response)
    except Exception as e:
        print(f"[ERROR] {e}")
        return (f"ERROR: {str(e)}", [])


def parse_model_response(text):
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


def analysis_loop():
    global is_analyzing
    is_analyzing = True
    print("[ANALYSIS] Starting analysis loop")

    while is_analyzing:
        with current_frame_lock:
            if current_frame is None:
                time.sleep(0.1)
                continue
            frame_data = current_frame

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[ANALYSIS] Analyzing frame at {timestamp}")

        summary, objects = analyze_frame(frame_data)
        print(f"[ANALYSIS] Result: {summary}")

        entry = ContextEntry(
            image_data=base64.b64encode(frame_data).decode("utf-8"),
            summary=summary,
            objects=objects,
            timestamp=timestamp
        )

        with lock:
            context_log.append(entry)
            if len(context_log) > max_context_length:
                del context_log[:len(context_log) - max_context_length]

        time.sleep(capture_interval_ms / 1000.0)

    print("[ANALYSIS] Stopped analysis loop")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/frame')
def api_frame():
    with current_frame_lock:
        if current_frame:
            return f"data:image/jpeg;base64,{base64.b64encode(current_frame).decode()}"
    return ""


@app.route('/api/context', methods=['GET', 'POST'])
def api_context():
    global capture_interval_ms, is_analyzing, analysis_thread

    if request.method == 'POST':
        data = request.json or {}
        action = data.get('action', '')
        interval = data.get('interval')

        if interval:
            capture_interval_ms = int(interval)
            print(f"[CONFIG] Interval set to {capture_interval_ms}ms")

        if action == 'start':
            if not is_analyzing:
                is_analyzing = True
                analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
                analysis_thread.start()
            print("[API] Analysis started")

        elif action == 'stop':
            is_analyzing = False
            print("[API] Analysis stopped")

        elif action == 'clear':
            with lock:
                del context_log[:]
            print("[API] Context cleared")

    with lock:
        log_data = [{
            "timestamp": e.timestamp,
            "summary": e.summary,
            "objects": e.objects,
            "image": e.image_data
        } for e in context_log]

    return jsonify({
        "interval": capture_interval_ms,
        "is_running": is_analyzing,
        "context": log_data
    })


if __name__ == "__main__":
    import atexit

    def cleanup():
        global is_capturing, is_analyzing
        is_capturing = False
        is_analyzing = False
        print("\n[EXIT] Shutting down...")

    atexit.register(cleanup)

    capture_thread = threading.Thread(target=capture_frame, daemon=True)
    capture_thread.start()

    print("=" * 60)
    print("Live Video Reasoning Web App")
    print("=" * 60)
    print("Open http://localhost:5001 in your browser")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)