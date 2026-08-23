"""Lightweight face liveness and anti-spoofing for FaceTrack Live."""

from __future__ import annotations

import hashlib
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import face_recognition
import numpy as np

MODEL_URL = (
    "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/"
    "MiniFASNetV2.onnx"
)
MODEL_SHA256 = "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907"
MODEL_SIZE = (80, 80)


@dataclass(frozen=True)
class LivenessResult:
    allowed: bool
    ai_real: bool
    blink_verified: bool
    ai_score: float | None
    message: str


class LivenessGuard:
    """Stateful AI anti-spoof + active blink challenge with low CPU overhead."""

    def __init__(self, challenge_timeout: float = 12.0, spoof_refresh: float = 1.5) -> None:
        self.challenge_timeout = challenge_timeout
        self.spoof_refresh = spoof_refresh
        self.challenge_started = time.monotonic()
        self.blink_count = 0
        self._eye_was_closed = False
        self._last_ai_check = 0.0
        self._last_ai_real = False
        self._last_ai_score: float | None = None
        self._ai_error: str | None = None
        self._session = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._last_face_location: tuple[int, int, int, int] | None = None
        self._last_face_seen = 0.0
        self._missing_face_grace = 1.5
        self._live_confirmations = 0

    def reset(self) -> None:
        self.challenge_started = time.monotonic()
        self.blink_count = 0
        self._eye_was_closed = False
        self._last_ai_check = 0.0
        self._last_ai_real = False
        self._last_ai_score = None
        self._ai_error = None
        self._last_face_location = None
        self._last_face_seen = 0.0
        self._live_confirmations = 0

    @property
    def blink_verified(self) -> bool:
        return self.blink_count >= 1

    @property
    def recent_face_location(self) -> tuple[int, int, int, int] | None:
        if self._last_face_location is None:
            return None
        if time.monotonic() - self._last_face_seen > self._missing_face_grace:
            return None
        return self._last_face_location

    def evaluate(self, frame: np.ndarray, location: tuple[int, int, int, int]) -> LivenessResult:
        if time.monotonic() - self.challenge_started > self.challenge_timeout:
            self.reset()
            return LivenessResult(False, False, False, None, "Liveness timed out — please blink again.")

        self._last_face_location = location
        self._last_face_seen = time.monotonic()
        top, right, bottom, left = location
        height, width = frame.shape[:2]
        top = max(0, int(top)); right = min(width, int(right))
        bottom = min(height, int(bottom)); left = max(0, int(left))
        if right <= left or bottom <= top:
            return LivenessResult(False, False, self.blink_verified, self._last_ai_score, "Face is out of frame.")

        if not self.blink_verified:
            self._update_blink(frame, location)

        self._update_ai_spoof(frame, (top, right, bottom, left))

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
        if self._live_confirmations < 2:
            return LivenessResult(False, True, True, self._last_ai_score,
                                  "Blink detected ✓ Verifying live face…")
        return LivenessResult(True, True, True, self._last_ai_score, "Liveness verified.")

    def evaluate_missing_face(self, frame: np.ndarray) -> LivenessResult:
        location = self.recent_face_location
        if location is None:
            return LivenessResult(False, False, self.blink_verified, self._last_ai_score, "No face detected")
        if not self.blink_verified:
            self._update_blink(frame, location)
        if self._ai_error:
            return LivenessResult(False, False, self.blink_verified, self._last_ai_score,
                                  f"Anti-spoofing unavailable: {self._ai_error}")
        if not self._last_ai_real:
            return LivenessResult(False, False, self.blink_verified, self._last_ai_score,
                                  "Possible photo/screen detected — use your real face.")
        if not self.blink_verified:
            remaining = max(0.0, self.challenge_timeout - (time.monotonic() - self.challenge_started))
            return LivenessResult(False, True, False, self._last_ai_score,
                                  f"Keep face steady. Blink once ({remaining:.0f}s).")
        if self._live_confirmations < 2:
            return LivenessResult(False, True, True, self._last_ai_score,
                                  "Blink detected ✓ Verifying live face…")
        return LivenessResult(False, True, True, self._last_ai_score, "Blink detected ✓ Keep face steady.")

    def _ensure_model(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        model_dir = Path.home() / ".facetrack" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "MiniFASNetV2.onnx"
        if not model_path.exists() or not self._valid_model(model_path):
            temp_path = model_path.with_suffix(".download")
            urllib.request.urlretrieve(MODEL_URL, temp_path)
            if not self._valid_model(temp_path):
                temp_path.unlink(missing_ok=True)
                raise RuntimeError("Downloaded anti-spoofing model failed its integrity check.")
            temp_path.replace(model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path.as_posix(), sess_options=options, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    @staticmethod
    def _valid_model(path: Path) -> bool:
        if not path.exists() or path.stat().st_size < 100_000:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == MODEL_SHA256

    def _update_ai_spoof(self, frame: np.ndarray, location: tuple[int, int, int, int]) -> None:
        now = time.monotonic()
        if now - self._last_ai_check < self.spoof_refresh:
            return
        self._last_ai_check = now
        self._ai_error = None
        try:
            self._ensure_model()
            assert self._session is not None and self._input_name is not None and self._output_name is not None
            top, right, bottom, left = location
            src_h, src_w = frame.shape[:2]
            box_w = max(1, right - left)
            box_h = max(1, bottom - top)
            # Match the upstream MiniFASNetV2 inference exactly: scale is capped
            # by the available image area, then resize to the model input.
            scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, 2.7)
            new_w = box_w * scale
            new_h = box_h * scale
            center_x = left + box_w / 2.0
            center_y = top + box_h / 2.0
            x1 = max(0, int(center_x - new_w / 2.0))
            y1 = max(0, int(center_y - new_h / 2.0))
            x2 = min(src_w - 1, int(center_x + new_w / 2.0))
            y2 = min(src_h - 1, int(center_y + new_h / 2.0))
            cropped = frame[y1:y2 + 1, x1:x2 + 1]
            if cropped.size == 0:
                self._last_ai_real = False
                self._live_confirmations = 0
                return
            input_cfg = self._session.get_inputs()[0]
            input_size = tuple(input_cfg.shape[2:])
            face = cv2.resize(cropped, input_size[::-1], interpolation=cv2.INTER_LINEAR)
            # The upstream model expects raw BGR float32 values, NOT /255 normalization.
            tensor = np.transpose(face.astype(np.float32), (2, 0, 1))[None, ...]
            logits = np.asarray(self._session.run([self._output_name], {self._input_name: tensor})[0], dtype=np.float32)
            logits -= np.max(logits, axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
            label_idx = int(np.argmax(probabilities[0]))
            score = float(probabilities[0, label_idx])
            self._last_ai_score = float(probabilities[0, 1])
            self._last_ai_real = label_idx == 1 and score >= 0.65
            if self._last_ai_real:
                self._live_confirmations += 1
            else:
                self._live_confirmations = 0
        except Exception as error:
            self._last_ai_real = False
            self._live_confirmations = 0
            self._last_ai_score = None
            self._ai_error = str(error)

    def _update_blink(self, frame: np.ndarray, location: tuple[int, int, int, int]) -> None:
        scale = 0.5
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small_location = tuple(int(value * scale) for value in location)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        landmarks = face_recognition.face_landmarks(rgb, [small_location])
        if not landmarks:
            return
        eyes = landmarks[0]
        left_eye = eyes.get("left_eye"); right_eye = eyes.get("right_eye")
        if not left_eye or not right_eye:
            return
        ear = (self._eye_aspect_ratio(left_eye) + self._eye_aspect_ratio(right_eye)) / 2.0
        closed = ear < 0.22
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
