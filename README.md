# Local Live Video Reasoning System v2.0

A robust prototype that captures live video from your Mac webcam, analyzes frames using a local Ollama vision model, and maintains layered memory of scene observations. All processing happens locally on your machine - no paid APIs required.

## Version 2.0 What's New

- **Layered Memory Architecture** - Current state, recent context, and event history
- **Detailed Summaries** - Rich, memory-usable scene descriptions
- **Multi-layered Change Detection** - Summary, objects, actions, and status changes
- **Robust Error Handling** - Retries, timeouts, graceful degradation
- **Modular Code** - Separate concerns: camera, vision, memory, state_manager
- **Detailed Output Files** - current_state.md, event_log.md, state.json, recent_context.md, agent_summary.md

## What It Does

1. **Captures continuous video** from your Mac's built-in webcam
2. **Samples frames** at a configurable interval (default: every 2 seconds)
3. **Sends frames to a local Ollama vision model** (like llava)
4. **Analyzes the scene** with detailed structured output
5. **Maintains layered memory** - current, recent context, and event history
6. **Detects meaningful changes** with anti-duplication logic
7. **Writes structured state files** for both humans and machines

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Camera   │────>│   Vision   │────>│   Memory  │
│  Module   │     │  Module   │     │  Module   │
└─────────────┘     └──────────────┘     └─────────────┘
                                            │
                                            ▼
                    ┌──────────────┐     ┌─────────────┐
                    │   State    │<────│   Parser   │
                    │  Manager  │     │           │
                    └──────────────┘     └─────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
   current_state.md   event_log.md   state.json
          │                │                │
          ▼                ▼                ▼
   recent_context.json  agent_summary.md
```

## Requirements

- macOS (tested on Mac)
- Python 3.8+
- OpenCV for Python (`opencv-python`)
- `requests` library
- Ollama installed locally
- A vision-capable Ollama model

## Installation

### 1. Install Python Dependencies

```bash
pip install opencv-python requests
```

### 2. Install Ollama

```bash
brew install ollama
```

### 3. Pull a Vision Model

```bash
ollama pull llava
```

### 4. Start Ollama

```bash
ollama serve
```

Or just start the app normally - Ollama runs in the background.

## Running the Prototype

```bash
python main.py
```

### Controls

- **Press `q`** to quit the application
- The live webcam feed displays in a window
- Terminal shows all analysis results and state updates

## File Structure

```
AIBOT/
├── main.py              # Entry point and main loop
├── config.py            # All configuration constants
├── camera.py           # Webcam capture with reliability
├── vision.py            # Ollama API and parsing
├── memory.py            # Layered memory system
├── state_manager.py     # State file writing
├── prompts.py          # Vision model prompts
├── utils.py           # Utilities and similarity functions
├── outputs/            # Output files directory
│   ├── current_state.md
│   ├── event_log.md
│   ├── state.json
│   ├── recent_context.md
│   ├── recent_context.json
│   └── agent_summary.md
```

## Configuration

Edit the **config.py** file to customize:

```python
WEBCAM_INDEX = 0                    # 0 = default Mac webcam
WEBCAM_ADDITIONAL_INDICES = [1, 2]  # Fallback camera indices
FRAME_SAMPLE_INTERVAL = 2             # seconds between captures
VISION_MODEL = "llava"               # Ollama vision model
MODEL_REQUEST_TIMEOUT = 90           # seconds
MODEL_MAX_RETRIES = 3                 # retry attempts
OUTPUT_FOLDER = Path("outputs")        # where to write files
MAX_RECENT_CONTEXT = 10                 # observations to keep
MIN_SIMILARITY_THRESHOLD = 0.6         # for change detection
```

## Output Files

### current_state.md

Detailed human-readable current state:

```markdown
# Current State

**Timestamp:** 2026-04-18T23:30:00
**Status:** Active

## Current Scene

A person is sitting at a desk facing the camera. They have dark hair and are wearing a blue shirt. The desk has a laptop, a coffee cup, and a phone. Natural light from a window on the left illuminates the scene.

## Visible Objects

person, desk, laptop, coffee cup, phone, window, chair

## Detected Actions

sitting, looking, typing

## Scene Status: active

## Confidence: 0.85

## Last Change

Objects: +person

## Event Count: 5

## Agent Summary

A person is working at their desk with a laptop, coffee, and phone visible. They are actively using the computer.
```

### event_log.md

Append-only log of meaningful events:

```markdown
# Event Log

### 2026-04-18T23:25:00

**scene_change**: Scene changed: A person sitting at desk facing camera with laptop.

---

### 2026-04-18T23:30:00

**objects_change**: Objects: +coffee cup, +phone
```

### state.json

Machine-readable structured state:

```json
{
  "timestamp": "2026-04-18T23:30:00",
  "current": {
    "summary": "A person sitting at desk...",
    "objects": ["person", "desk", "laptop", "phone"],
    "actions": ["sitting", "typing"],
    "scene_status": "active",
    "confidence": 0.85
  },
  "previous": {
    "summary": "Empty desk with laptop",
    "objects": ["desk", "laptop"],
    "actions": []
  },
  "event_count": 5,
  "recent_context": [...],
  "agent_summary": "A person is working..."
}
```

### recent_context.json

Recent observations for memory:

```json
{
  "timestamp": "2026-04-18T23:30:00",
  "recent_context": [
    {"timestamp": "...", "summary": "...", "objects": [...], "actions": [...]},
    {"timestamp": "...", "summary": "...", "objects": [...], "actions": [...]}
  ],
  "agent_summary": "..."
}
```

### agent_summary.md

LLM-generated summary for reasoning:

```markdown
# Agent Summary

**Last Updated:** 2026-04-18T23:30:00

A single person is actively working at their desk. They arrived about 5 minutes ago and have been using their laptop continuously. Coffee and phone are on the desk. No significant changes in the environment.
```

## Change Detection

The system uses multiple layers of change detection:

1. **Summary Similarity** - Word overlap + sequence matching + keyword analysis
2. **Object Diff** - Added/removed objects
3. **Action Diff** - Changed activities
4. **Status Change** - Active/idle transitions
5. **Anti-duplication** - Won't log similar observations repeatedly

## Memory Architecture

### Layer A: Current State
What's happening now - used for immediate context

### Layer B: Recent Context
Last 10 observations - provides rolling memory

### Layer C: Event History
Only meaningful events - stored long-term

### Layer D: Agent Summary
LLM-generated summary for reasoning (future feature)

## Robustness Features

- **Camera**: Multiple attempts, fallback indices, warmup frames
- **Vision**: 3 retries with exponential backoff, connection checking
- **Parsing**: JSON first, structured text fallback
- **Errors**: Graceful degradation, doesn't crash easily
- **Logging**: Detailed terminal output + optional file logging

## Future-Ready Architecture

The system is designed for a second reasoning layer:

1. Vision model extracts structured data
2. recent_context.json provides memory
3. agent_summary.md enables reasoning
4. Easy to add second LLM for higher-level reasoning

## Example Workflow

1. **Start Ollama**: `ollama serve`
2. **Run the script**: `python main.py`
3. **Watch** your terminal fill with detailed analysis
4. **Move around** in front of your camera
5. **Notice** changes detected and logged with context
6. **Check** the generated files in `outputs/`
7. **Press `q`** to quit

## Cleanup

The script creates temporary files which are cleaned up on exit. To remove all generated files:

```bash
rm -rf outputs/
mkdir outputs
```

## Limitations

- **Speed**: Depends on your hardware and model size
- **Model quality**: Local models are less capable than cloud APIs
- **No audio**: Video-only
- **Single camera**: Extended support would require restructuring
- **Basic change detection**: Word-overlap similarity is primitive compared to embeddings

## Common Issues

### "Cannot connect to Ollama"
- Make sure Ollama is running: `ollama serve`
- Check Ollama is installed: `ollama --version`

### "Cannot open webcam"
- Close other apps using the camera (FaceTime, Zoom, etc.)
- Check webcam index in config.py

### "Model not found"
- Pull the model: `ollama pull llava`
- Check available models: `ollama list`

### "Analysis failing"
- Check model is vision-capable (llava works, gemma4 does not)
- Check your system's RAM availability

## License

MIT - Use freely for learning and experimentation.