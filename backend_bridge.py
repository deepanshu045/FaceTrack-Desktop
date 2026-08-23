"""HTTP adapter between the desktop camera app and the deployed FaceTrack backend."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from urllib.parse import quote

import numpy as np
import requests


DEFAULT_BACKEND_URL = os.getenv("FACE_ATTENDANCE_BACKEND_URL", "https://facetrack-ggbe.onrender.com").rstrip("/")


@dataclass(frozen=True)
class RegisteredStudent:
    id: int
    roll_no: str
    name: str
    department: str
    encoding: np.ndarray


@dataclass(frozen=True)
class CollegeOption:
    name: str
    slug: str


@dataclass(frozen=True)
class RecognitionSettings:
    confidence_threshold: int
    distance_threshold: float
    sound_alerts: bool


def _api_url(path: str) -> str:
    return f"{DEFAULT_BACKEND_URL}{path}" if path.startswith("/") else f"{DEFAULT_BACKEND_URL}/{path}"


def _request_json(method: str, path: str, **kwargs) -> dict:
    response = requests.request(method, _api_url(path), timeout=20, **kwargs)
    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text}

    if response.status_code >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        raise RuntimeError(detail or f"Request failed with status {response.status_code}.")
    return payload if isinstance(payload, dict) else {}


def get_college_slug() -> str:
    return os.getenv("FACE_ATTENDANCE_COLLEGE_SLUG", "legacy-college").strip().lower()


def list_active_colleges() -> list[CollegeOption]:
    payload = _request_json("get", "/public/colleges")
    colleges = payload.get("colleges", [])
    return [CollegeOption(name=row["name"], slug=row["slug"]) for row in colleges]


class AttendanceRepository:
    def __init__(self, access_code: str) -> None:
        self._access_code = access_code.strip()
        self._college_slug = ""
        self._college_id = 0
        self._recognition_settings = RecognitionSettings(85, 0.60, True)
        self._students: list[RegisteredStudent] = []
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._resolve_college()
        self._start_heartbeat()

    def _resolve_college(self) -> None:
        if not self._access_code:
            raise RuntimeError("Enter the camera access code.")

        payload = _request_json(
            "get",
            f"/public/college/access-code/{quote(self._access_code, safe='')}",
        )
        self._college_slug = payload["college_slug"]
        self._college_id = int(payload["college_id"])
        settings = payload.get("recognition_settings") or {}
        self._recognition_settings = RecognitionSettings(
            confidence_threshold=int(settings.get("confidence_threshold", 85)),
            distance_threshold=float(settings.get("distance_threshold", 0.60)),
            sound_alerts=bool(settings.get("sound_alerts", True)),
        )

        students = []
        for row in payload.get("students", []):
            encoding = row.get("face_encoding")
            if not encoding:
                continue
            students.append(
                RegisteredStudent(
                    id=int(row["id"]),
                    roll_no=str(row["roll_no"]),
                    name=str(row["name"]),
                    department=str(row.get("department") or "Not set"),
                    encoding=np.asarray(encoding, dtype=np.float64),
                )
            )

        self._students = students
        if not self._students:
            raise RuntimeError("No registered face encodings were found for this college.")

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="facetrack-desktop-heartbeat",
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.is_set():
            try:
                _request_json(
                    "post",
                    "/recognition/desktop-heartbeat",
                    json={"access_code": self._access_code},
                )
            except Exception:
                # Attendance must continue to work even when the status endpoint
                # is temporarily unavailable. The next heartbeat will retry.
                pass
            self._heartbeat_stop.wait(10)

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()

    @property
    def college_slug(self) -> str:
        return self._college_slug

    def recognition_settings(self) -> RecognitionSettings:
        return self._recognition_settings

    def students_with_faces(self) -> list[RegisteredStudent]:
        return list(self._students)

    def mark_present(self, student_id: int) -> dict:
        return _request_json(
            "post",
            f"/public/college/{quote(self._college_slug, safe='')}/mark-attendance",
            json={"student_id": student_id},
        )
