"""
Vision module for the Live Video Reasoning System.
Handles Ollama API communication with retries and robust parsing.
"""

import base64
import requests
import logging
import time
from typing import Optional, Tuple, Dict
from pathlib import Path

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


class VisionError(Exception):
    """Custom exception for vision errors."""
    pass


class OllamaConnectionError(VisionError):
    """Connection to Ollama failed."""
    pass


class OllamaModelError(VisionError):
    """Model not found or unavailable."""
    pass


class OllamaTimeoutError(VisionError):
    """Request timed out."""
    pass


class VisionClient:
    """Robust Ollama vision client with retries."""

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.model = model or config.VISION_MODEL
        self.timeout = config.MODEL_REQUEST_TIMEOUT
        self.max_retries = config.MODEL_MAX_RETRIES
        self.retry_backoff = config.MODEL_RETRY_BACKOFF
        self.rate_limit_delay = config.MODEL_RATE_LIMIT_DELAY
        self.log_raw = config.LOG_RAW_RESPONSES

        self.last_request_time = 0
        self.request_count = 0

    def check_connection(self) -> Tuple[bool, str]:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()

            available_models = [m.get("name", "") for m in data.get("models", [])]

            if self.model not in available_models:
                return False, f"Model '{self.model}' not found. Available: {available_models}"

            return True, "connected"

        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to Ollama. Is it running?"
        except requests.exceptions.Timeout:
            return False, "Connection timeout"
        except Exception as e:
            return False, f"Connection error: {str(e)}"

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image as base64."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    def _send_request(self, prompt: str, image_data: str) -> Tuple[Optional[dict], str]:
        """Send request to Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_data],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }

        try:
            self.request_count += 1
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            if self.log_raw:
                logger.debug(f"Raw response: {result}")

            return result, "success"

        except requests.exceptions.ConnectionError as e:
            return None, f"Connection error: {str(e)}"
        except requests.exceptions.Timeout as e:
            return None, f"Timeout: {str(e)}"
        except requests.exceptions.HTTPError as e:
            return None, f"HTTP error: {str(e)}"
        except Exception as e:
            return None, f"Request error: {str(e)}"

    def analyze_image(self, image_path: str, prompt: str) -> Tuple[Optional[Dict], str]:
        """Analyze image with retries."""
        self._apply_rate_limit()

        image_data = self._encode_image(image_path)
        if not image_data:
            return None, "Failed to encode image"

        last_error = "Unknown error"

        for attempt in range(self.max_retries):
            if attempt > 0:
                delay = self.retry_backoff ** attempt
                logger.info(f"Retry {attempt + 1}/{self.max_retries} after {delay:.1f}s...")
                time.sleep(delay)

            result, status = self._send_request(prompt, image_data)

            if result:
                response_text = result.get("response", "")
                if response_text:
                    parsed, parse_status = utils.parse_model_response(response_text)
                    if parsed:
                        logger.debug(f"Analysis successful (attempt {attempt + 1})")
                        return parsed, "success"

                    logger.warning(f"Parse failed: {parse_status}")
                    last_error = f"Parse failed: {parse_status}"
                    continue

                last_error = "Empty response"
            else:
                last_error = status
                logger.warning(f"Request failed: {status}")

        return None, f"All retries failed: {last_error}"

    def analyze_with_context(self, image_path: str, prompt: str,
                           previous_summary: str = "",
                           previous_objects: list = None,
                           previous_actions: list = None,
                           frame_count: int = 0) -> Tuple[Optional[Dict], str]:
        """Analyze image with previous context."""
        if previous_objects is None:
            previous_objects = []
        if previous_actions is None:
            previous_actions = []

        full_prompt = prompt.format(
            previous_summary=previous_summary,
            previous_objects=previous_objects,
            previous_actions=previous_actions,
            frame_count=frame_count
        )

        return self.analyze_image(image_path, full_prompt)

    def get_stats(self) -> dict:
        """Get client statistics."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "request_count": self.request_count,
            "last_request_time": self.last_request_time
        }


def create_vision_client(base_url: str = None, model: str = None) -> VisionClient:
    """Create and validate a vision client."""
    client = VisionClient(base_url, model)

    connected, msg = client.check_connection()
    if not connected:
        logger.warning(f"Vision client check: {msg}")

    return client