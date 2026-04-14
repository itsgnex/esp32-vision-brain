# Local Live Video Reasoning System

A prototype that captures live video from your Mac webcam, analyzes frames using a local Ollama vision model, and maintains rolling memory of scene observations. All processing happens locally on your machine - no paid APIs required.

## What It Does

1. **Captures continuous video** from your Mac's built-in webcam
2. **Samples frames** at a configurable interval (default: every 2 seconds)
3. **Sends frames to a local Ollama vision model** (like llava)
4. **Analyzes the scene** and extracts summary + visible objects
5. **Maintains rolling memory** - detects meaningful changes over time
6. **Writes state files** - current_state.md, event_log.md, state.json
7. **Prints live updates** to your terminal

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   OpenCV    │────>│   Sampler   │────>│   Ollama    │
│  Webcam     │     │  (perodic)  │     │  Vision    │
└─────────────┘     └──────────────┘     └─────────────┘
                                              │
                                              v
                   ┌──────────────┐     ┌─────────────┐
                   │   State      │<────│   Parser    │
                   │   Manager    │     │             │
                   └──────────────┘     └─────────────┘
                          │
           ┌──────────────┼──────────────┐
           v              v              v
    current_state.md  event_log.md   state.json
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
# Option 1: llava (recommended for starting)
ollama pull llava

# Option 2: llava with mistral base
ollama pull llava:mistral

# Option 3: llava 34b (better quality, requires more RAM)
ollama pull llava:34b

# List available vision models
ollama list | grep -i vision
```

### 4. Start Ollama

```bash
# In a separate terminal, run:
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

## Configuration

Edit the **CONFIGURATION SECTION** at the top of `main.py` to customize:

```python
WEBCAM_INDEX = 0              # 0 = default Mac webcam
FRAME_SAMPLE_INTERVAL = 2     # seconds between captures
VISION_MODEL = "llava:7b"     # Ollama vision model
OUTPUT_FOLDER = Path(".")     # where to write files
TERMINAL_LOG = True           # print to terminal
```

## Output Files

### `current_state.md`

Human-readable current state:

```markdown
# Current State

**Timestamp:** 2026-01-13T10:30:00

**Status:** Active

## Current Scene

Living room with desk and chair, person sitting at desk facing camera.

## Visible Objects

person, desk, chair, laptop, window

## Last Change

scene changed: living room with desk and chair, person sitting at desk facing camera.

## Event Count: 5
```

### `event_log.md`

Append-only log of meaningful changes:

```markdown
# Event Log

### 2026-01-13T10:25:00

scene changed: empty room with desk and window

---

### 2026-01-13T10:30:00

objects: +person, -none
```

### `state.json`

Machine-readable structured state:

```json
{
  "timestamp": "2026-01-13T10:30:00.123456",
  "latest_summary": "Living room with desk and chair, person sitting at desk facing camera.",
  "objects": ["person", "desk", "chair", "laptop", "window"],
  "last_event": "objects: +person, -none",
  "event_count": 5,
  "history": [
    {"timestamp": "...", "summary": "...", "change": "..."}
  ]
}
```

## Example Workflow

1. **Start Ollama** in a separate terminal: `ollama serve`
2. **Run the script**: `python main.py`
3. **Watch** your terminal fill with live analysis
4. **Move around** in front of your camera
5. **Notice** changes detected and logged
6. **Check** the generated markdown/JSON files
7. **Press `q`** to quit

## Limitations

- **Speed**: Depends on your hardware and model size
- **Model quality**: Local models are less capable than cloud APIs
- **No audio**: Video-only (add audio if you want speech analysis)
- **Single camera**: Extended support would require restructuring
- **Basic change detection**: Word-overlap similarity is primitive

## Common Issues

### "Cannot connect to Ollama"
- Make sure Ollama is running: `ollama serve` or start the app
- Check Ollama is installed: `ollama --version`

### "Cannot open webcam"
- Close other apps using the camera ( FaceTime, Zoom, etc.)
- Check webcam index: try `WEBCAM_INDEX = 1`

### "Model not found"
- Pull the model: `ollama pull llava`
- Check available models: `ollama list`

## Suggestions for Next Improvements

1. **Faster sampling** - reduce interval for more real-time feel
2. **Multiple cameras** - add support for external webcams
3. **Audio integration** - add speech-to-text for conversation
4. **Better prompts** - improve structured output parsing
5. **Web dashboard** - serve current state over HTTP
6. **Event webhooks** - notify on significant changes
7. **Database storage** - use SQLite instead of JSON for history
8. **Better similarity** - use embeddings for change detection

## Cleanup

The script creates a temporary `.temp_frame.jpg` file which is cleaned up on exit. To remove all generated files:

```bash
rm current_state.md event_log.md state.json .temp_frame.jpg
```

## License

MIT - Use freely for learning and experimentation.