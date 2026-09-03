"""HTTP adapter between the desktop camera app and the FaceTrack FastAPI backend."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import quote

import numpy as np
import requests


logger = logging.getLogger(__name__)
DEFAULT_BACKEND_URL = "https://facetrack-ggbe.onrender.com"


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
class ActiveLecture:
    id: int
    subject: str
    date: str
    start_time: str
    end_time: str
    class_name: str
    section: str
    department: str


@dataclass(frozen=True)
class RecognitionSettings:
    confidence_threshold: int
    distance_threshold: float
    sound_alerts: bool


def _api_url(path: str) -> str:
    return f"{DEFAULT_BACKEND_URL}{path}" if path.startswith("/") else f"{DEFAULT_BACKEND_URL}/{path}"


def _request_json(method: str, path: str, **kwargs) -> dict:
    url = _api_url(path)
    timeout = kwargs.pop("timeout", 30)
    logger.info("Backend request: %s %s", method.upper(), url)
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        logger.error("Backend connection failed: %s", exc)
        raise RuntimeError(
            f"Cannot connect to FaceTrack backend at {DEFAULT_BACKEND_URL}. "
            "Start the FastAPI server or set FACE_ATTENDANCE_BACKEND_URL."
        ) from exc

    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text}

    logger.info("Backend response: %s %s", response.status_code, response.url)
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
        self._attendance_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="attendance-api")
        self._attendance_lock = threading.Lock()
        self._attendance_inflight: set[tuple[int, int]] = set()
        self._attendance_last_submit: dict[tuple[int, int], float] = {}
        # Session-local attendance cache. A (student, lecture) pair is cached
        # after a successful submission, so repeated recognition frames do not
        # keep calling the attendance endpoint for the same lecture.
        self._attendance_cache: set[tuple[int, int]] = set()
        self._lecture_cache: dict[int, ActiveLecture | None] = {}
        self._lecture_cache_at: dict[int, float] = {}
        self._lecture_refreshing: set[int] = set()
        self._lecture_lock = threading.Lock()
        self._lecture_cache_ttl = 10.0
        self._lecture_refresh_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lecture-api")
        self._resolve_college()
        self._start_heartbeat()

    def _resolve_college(self) -> None:
        if not self._access_code:
            raise RuntimeError("Enter the camera access code.")
        payload = _request_json("get", f"/public/college/access-code/{quote(self._access_code, safe='')}")
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
            students.append(RegisteredStudent(
                id=int(row["id"]),
                roll_no=str(row["roll_no"]),
                name=str(row["name"]),
                department=str(row.get("department") or "Not set"),
                encoding=np.asarray(encoding, dtype=np.float64),
            ))
        self._students = students
        if not self._students:
            raise RuntimeError("No registered face encodings were found for this college.")

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name="facetrack-desktop-heartbeat")
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.is_set():
            try:
                _request_json("post", "/recognition/desktop-heartbeat", json={"access_code": self._access_code}, timeout=10)
            except Exception:
                pass
            self._heartbeat_stop.wait(15)

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        self._attendance_executor.shutdown(wait=False, cancel_futures=True)
        self._lecture_refresh_executor.shutdown(wait=False, cancel_futures=True)

    @property
    def college_slug(self) -> str:
        return self._college_slug

    def recognition_settings(self) -> RecognitionSettings:
        return self._recognition_settings

    def students_with_faces(self) -> list[RegisteredStudent]:
        return list(self._students)

    @staticmethod
    def _lecture_from_payload(payload: dict) -> ActiveLecture | None:
        row = payload.get("lecture")
        if not payload.get("active") or not row:
            return None
        return ActiveLecture(
            id=int(row["id"]),
            subject=str(row["subject"]),
            date=str(row["date"]),
            start_time=str(row["start_time"]),
            end_time=str(row["end_time"]),
            class_name=str(row.get("class_name") or ""),
            section=str(row.get("section") or ""),
            department=str(row.get("department") or ""),
        )

    def _fetch_active_lecture(self, student_id: int, timeout: float = 1.0) -> ActiveLecture | None:
        try:
            payload = _request_json(
                "get",
                f"/public/college/access-code/{quote(self._access_code, safe='')}/active-lecture/{student_id}",
                timeout=timeout,
            )
            lecture = self._lecture_from_payload(payload)
            with self._lecture_lock:
                self._lecture_cache[student_id] = lecture
                self._lecture_cache_at[student_id] = time.monotonic()
            return lecture
        except Exception as exc:
            logger.warning("Active lecture lookup failed for student %s: %s", student_id, exc)
            return None
        finally:
            with self._lecture_lock:
                self._lecture_refreshing.discard(student_id)

    def _refresh_lecture_background(self, student_id: int) -> None:
        with self._lecture_lock:
            if student_id in self._lecture_refreshing:
                return
            self._lecture_refreshing.add(student_id)
        try:
            self._lecture_refresh_executor.submit(self._fetch_active_lecture, student_id, 10.0)
        except RuntimeError:
            with self._lecture_lock:
                self._lecture_refreshing.discard(student_id)

    def active_lecture_for_student(self, student_id: int) -> ActiveLecture | None:
        """Return cached lecture immediately and refresh it in the background.

        This method is called from the Tkinter UI thread. It must never wait on
        the FastAPI server during normal recognition, otherwise a slow backend
        response makes the camera appear to freeze.
        """
        now = time.monotonic()
        with self._lecture_lock:
            cached = self._lecture_cache.get(student_id)
            cached_at = self._lecture_cache_at.get(student_id, 0.0)
            fresh = cached_at > 0 and now - cached_at < self._lecture_cache_ttl

        if not fresh:
            self._refresh_lecture_background(student_id)
        return cached

    def _submit_attendance(self, student_id: int, lecture_id: int | None) -> None:
        key = (student_id, lecture_id or 0)
        try:
            payload = {"student_id": student_id}
            if lecture_id is not None:
                payload["lecture_id"] = lecture_id
            _request_json(
                "post",
                f"/public/college/{quote(self._college_slug, safe='')}/mark-attendance",
                json=payload,
                timeout=10,
            )
            # Any successful 2xx response means the server accepted the request.
            # Whether it created a new row or reported an existing row, this
            # scanner does not need to submit the same student/lecture pair again.
            with self._attendance_lock:
                self._attendance_cache.add(key)
        except Exception as exc:
            logger.warning("Attendance submission failed for student %s: %s", student_id, exc)
        finally:
            with self._attendance_lock:
                self._attendance_inflight.discard(key)

    def mark_present(self, student_id: int, lecture_id: int | None = None) -> dict:
        """Queue attendance without blocking the camera/UI thread.

        Attendance is cached by (student_id, lecture_id) for the lifetime of
        this repository/session. Recognition can therefore continue every 0.5s
        without repeatedly submitting the same attendance event.
        """
        key = (student_id, lecture_id or 0)
        now = time.monotonic()
        with self._attendance_lock:
            if key in self._attendance_cache:
                return {"queued": False, "already_marked": True, "cached": True}
            last_submit = self._attendance_last_submit.get(key, 0.0)
            if key in self._attendance_inflight or now - last_submit < 10.0:
                return {"queued": False, "already_marked": True, "cached": False}
            self._attendance_inflight.add(key)
            self._attendance_last_submit[key] = now
        try:
            self._attendance_executor.submit(self._submit_attendance, student_id, lecture_id)
        except RuntimeError:
            with self._attendance_lock:
                self._attendance_inflight.discard(key)
            raise
        return {"queued": True, "already_marked": False, "cached": False}
