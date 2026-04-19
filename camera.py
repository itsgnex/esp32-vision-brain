"""
Camera module for the Live Video Reasoning System.
Provides reliable webcam capture with retries and validation.
"""

import cv2
import time
import logging
from typing import Optional
from pathlib import Path

# Import config
try:
    import config
except ImportError:
    from . import config

logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Custom exception for camera errors."""
    pass


class Camera:
    """Reliable webcam capture with validation."""

    def __init__(self, index: int = 0, additional_indices: list = None):
        self.index = index
        self.additional_indices = additional_indices or []
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.warmup_frames = config.WEBCAM_WARMUP_FRAMES
        self.retry_attempts = config.WEBCAM_RETRY_ATTEMPTS
        self.retry_delay = config.WEBCAM_RETRY_DELAY
        self.resolution = (config.WEBCAM_RESOLUTION_WIDTH, config.WEBCAM_RESOLUTION_HEIGHT)

    def open(self) -> bool:
        """Open camera with retries and fallback indices."""
        indices_to_try = [self.index] + self.additional_indices

        for idx in indices_to_try:
            logger.info(f"Attempting to open camera index {idx}...")

            for attempt in range(self.retry_attempts):
                try:
                    cap = cv2.VideoCapture(idx)

                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

                        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                        self.cap = cap
                        self.index = idx
                        logger.info(f"Camera opened successfully (index: {idx}, resolution: {actual_width}x{actual_height})")
                        return True

                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for index {idx}: {e}")

                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)

        logger.error(f"Failed to open camera after {self.retry_attempts} attempts")
        return False

    def read(self):
        """Read a frame from the camera."""
        if not self.cap or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        self.frame_count += 1

        if not ret:
            logger.warning(f"Failed to read frame #{self.frame_count}")
            return False, None

        return True, frame

    def read_valid(self):
        """Read a valid frame (not empty, not too dark)."""
        max_attempts = 10
        blank_threshold = 10

        for _ in range(max_attempts):
            ret, frame = self.read()

            if not ret:
                continue

            if frame is None:
                continue

            if frame.size == 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = gray.mean()

            if mean_brightness > blank_threshold:
                return True, frame

            logger.debug(f"Skipping dark/blank frame (brightness: {mean_brightness:.1f})")

        return False, None

    def warmup(self) -> int:
        """Warmup camera by reading frames."""
        if not self.cap:
            return 0

        warmup_count = 0
        frames_to_skip = self.warmup_frames

        logger.info(f"Warming up camera ({frames_to_skip} frames)...")

        for _ in range(frames_to_skip):
            ret, frame = self.read()
            if ret and frame is not None:
                warmup_count += 1

        logger.info(f"Warmup complete ({warmup_count} frames read)")
        return warmup_count

    def is_opened(self) -> bool:
        """Check if camera is open."""
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        """Release camera resources."""
        if self.cap:
            try:
                self.cap.release()
                logger.info("Camera released")
            except Exception as e:
                logger.warning(f"Error releasing camera: {e}")
            finally:
                self.cap = None

    def get_info(self) -> dict:
        """Get camera information."""
        if not self.cap or not self.cap.isOpened():
            return {"opened": False}

        return {
            "opened": True,
            "index": self.index,
            "resolution": self.resolution,
            "frame_count": self.frame_count
        }


def save_frame(frame, path: Path, quality: int = 90) -> bool:
    """Save frame to file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if success:
            logger.debug(f"Frame saved to {path}")
        return success
    except Exception as e:
        logger.error(f"Failed to save frame: {e}")
        return False


def create_display_window(name: str = "Live Video Reasoning", width: int = 640, height: int = 480):
    """Create a display window."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)


def show_frame(frame, name: str = "Live Video Reasoning", wait_ms: int = 1) -> int:
    """Show frame in window and return key press."""
    cv2.imshow(name, frame)
    return cv2.waitKey(wait_ms) & 0xFF


def destroy_windows() -> None:
    """Destroy all windows."""
    cv2.destroyAllWindows()


def check_camera_available() -> bool:
    """Check if any camera is available."""
    for i in range(4):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                return True
        except Exception:
            pass
    return False