"""
Configuration constants for the Live Video Reasoning System.
"""

from pathlib import Path

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================

WEBCAM_INDEX = 0
WEBCAM_ADDITIONAL_INDICES = [1, 2]
WEBCAM_RESOLUTION_WIDTH = 640
WEBCAM_RESOLUTION_HEIGHT = 480
WEBCAM_WARMUP_FRAMES = 5
WEBCAM_RETRY_ATTEMPTS = 3
WEBCAM_RETRY_DELAY = 1.0

# ============================================================================
# VISION MODEL CONFIGURATION
# ============================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
VISION_MODEL = "llava:latest"
FRAME_SAMPLE_INTERVAL = 2
MODEL_REQUEST_TIMEOUT = 90
MODEL_MAX_RETRIES = 3
MODEL_RETRY_BACKOFF = 2.0
MODEL_RATE_LIMIT_DELAY = 0.5
LOG_RAW_RESPONSES = False

# ============================================================================
# MEMORY CONFIGURATION
# ============================================================================

MAX_RECENT_CONTEXT = 10
MAX_EVENT_HISTORY = 50
MIN_SIMILARITY_THRESHOLD = 0.6
ACTION_SIMILARITY_THRESHOLD = 0.7

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

OUTPUT_FOLDER = Path(".")

TERMINAL_LOG = True
TERMINAL_VERBOSE = True

LOG_FILE = None
LOG_TO_FILE = False