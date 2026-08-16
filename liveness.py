"""Face liveness / anti-spoofing guard for FaceTrack Live.

The guard deliberately fails closed: a face is allowed to reach attendance
recognition only after both checks pass:

1. DeepFace's neural anti-spoofing model classifies the camera face as real.
2. The person completes an active blink challenge.

The anti-spoof model is loaded lazily because its first use downloads the
pre-trained weights. Subsequent checks reuse the loaded model/cache.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import face_recognition
import numpy as np


@dataclass(frozen=True)
class LivenessResult:
    allowed: bool
    ai_real: bool
    blink_verified: bool
    ai_score: float | None
    message: str


class LivenessGuard:
    """Stateful liveness challenge used by the recognition worker."""

    def __init__(self, challenge_timeout: float = 12.0, spoof_refresh: float = 0.8) -> None:
        self.challenge_timeout = challenge_timeout
        self.spoof_refresh = spoof_refresh
        self.challenge_started = time.monotonic()
        self.blink_count = 0
        self._eye_was_closed = False
        self._last_ai_check = 0.0
        self._last_ai_real = False
        self._last_ai_score: float | None = None
        self._ai_error: str | None = None
        self._deepface = None
        self._deepface_load_error: str | None = None

    def reset(self) -> None:
        self.challenge_started = time.monotonic()
        self.blink_count = 0
        self._eye_was_closed = False
        self._last_ai_check = 0.0
        self._last_ai_real = False
        self._last_ai_score = None
        self._ai_error = None

    @property
    def blink_verified(self) -> bool:
        return self.blink_count >= 1

    def evaluate(
        self,
        frame: np.ndarray,
        location: tuple[int, int, int, int],
    ) -> LivenessResult:
        """Evaluate one face. The input location is in the frame's current scale."""
        if time.monotonic() - self.challenge_started > self.challenge_timeout:
            self.reset()
            return LivenessResult(False, False, False, None, "Liveness timed out — please blink again.")

        top, right, bottom, left = location
        height, width = frame.shape[:2]
        top = max(0, int(top))
        right = min(width, int(right))
        bottom = min(height, int(bottom))
        left = max(0, int(left))
        if right <= left or bottom <= top:
            return LivenessResult(False, False, self.blink_verified, self._last_ai_score, "Face is out of frame.")

        # Use a little margin so the anti-spoof model sees the whole face region.
        pad_x = int((right - left) * 0.20)
        pad_y = int((bottom - top) * 0.25)
        crop = frame[max(0, top - pad_y):min(height, bottom + pad_y),
                     max(0, left - pad_x):min(width, right + pad_x)]
        if crop.size == 0:
            return LivenessResult(False, False, self.blink_verified, self._last_ai_score, "Unable to read face.")

        self._update_blink(crop)
        self._update_ai_spoof(crop)

        if self._ai_error:
            return LivenessResult(False, False, self.blink_verified, self._last_ai_score,
                                  f"Anti-spoofing unavailable: {self._ai_error}")
        if not self._last_ai_real:
            return LivenessResult(False, False, self.blink_verified, self._last_ai_score,
                                  "Possible photo/screen detected — use your real face.")
        if not self.blink_verified:
            remaining = max(0.0, self.challenge_timeout - (time.monotonic() - self.challenge_started))
            return LivenessResult(False, True, False, self._last_ai_score,
                                  f"Live face detected. Blink once ({remaining:.0f}s).")
        return LivenessResult(True, True, True, self._last_ai_score, "Liveness verified.")

    def _update_ai_spoof(self, crop: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_ai_check < self.spoof_refresh:
            return
        self._last_ai_check = now
        self._ai_error = None
        try:
            if self._deepface is None:
                from deepface import DeepFace
                self._deepface = DeepFace
            faces = self._deepface.extract_faces(
                img_path=crop,
                detector_backend="opencv",
                enforce_detection=False,
                align=True,
                anti_spoofing=True,
            )
            if not faces:
                self._last_ai_real = False
                self._last_ai_score = 0.0
                return
            face = faces[0]
            self._last_ai_real = bool(face.get("is_real", False))
            raw_score = face.get("antispoof_score", face.get("anti_spoofing_score"))
            self._last_ai_score = float(raw_score) if raw_score is not None else None
        except Exception as error:
            self._last_ai_real = False
            self._last_ai_score = None
            self._ai_error = str(error)

    def _update_blink(self, crop: np.ndarray) -> None:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        if len(locations) != 1:
            return
        landmarks = face_recognition.face_landmarks(rgb, locations)
        if not landmarks:
            return
        eyes = landmarks[0]
        left_eye = eyes.get("left_eye")
        right_eye = eyes.get("right_eye")
        if not left_eye or not right_eye:
            return
        ear = (self._eye_aspect_ratio(left_eye) + self._eye_aspect_ratio(right_eye)) / 2.0
        # The threshold is intentionally conservative; the challenge is only one blink.
        closed = ear < 0.20
        if closed:
            self._eye_was_closed = True
        elif self._eye_was_closed:
            self.blink_count += 1
            self._eye_was_closed = False

    @staticmethod
    def _eye_aspect_ratio(points: list[tuple[int, int]]) -> float:
        if len(points) < 6:
            return 1.0
        p = np.asarray(points[:6], dtype=np.float32)
        vertical_1 = np.linalg.norm(p[1] - p[5])
        vertical_2 = np.linalg.norm(p[2] - p[4])
        horizontal = np.linalg.norm(p[0] - p[3])
        return float((vertical_1 + vertical_2) / max(2.0 * horizontal, 1e-6))
