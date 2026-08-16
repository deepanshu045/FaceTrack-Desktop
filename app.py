"""FaceTrack Live - live face attendance with anti-spoofing and liveness."""

from __future__ import annotations

import time
import tkinter as tk
import winsound
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock, Thread
from tkinter import messagebox, ttk

import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageTk

from backend_bridge import AttendanceRepository, RecognitionSettings, RegisteredStudent
from liveness import LivenessGuard

RESOLUTIONS = {"480p": (640, 480), "720p": (1280, 720), "1080p": (1920, 1080)}
RecognitionResult = tuple[list[tuple[int, int, int, int]], int | None, float | None, bool, str]


class FaceAttendanceApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FaceTrack Live Attendance")
        self.geometry("1180x760")
        self.minsize(980, 650)
        self.configure(bg="#101820")
        self.camera: cv2.VideoCapture | None = None
        self.capture_thread: Thread | None = None
        self.capture_stop = Event()
        self.frame_lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.recognizer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face-recognition")
        self.recognition_future: Future[RecognitionResult] | None = None
        self.next_recognition_at = 0.0
        self.latest_recognition: RecognitionResult | None = None
        self.scan_token = 0
        self.repository: AttendanceRepository | None = None
        self.students: list[RegisteredStudent] = []
        self.last_match_id: int | None = None
        self.last_match_at = 0.0
        self.running = False
        self.liveness = LivenessGuard()
        self.camera_index = tk.IntVar(value=0)
        self.resolution = tk.StringVar(value="720p")
        self.target_fps = tk.StringVar(value="30")
        self.threshold = tk.DoubleVar(value=0.50)
        self.sound_enabled = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Ready. Start scanning when the camera is available.")
        self.person = tk.StringVar(value="No face detected")
        self.liveness_status = tk.StringVar(value="Liveness: waiting")
        self.access_code = tk.StringVar()
        self.selected_college_slug = ""
        self._build_college_access_page()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build_college_access_page(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("College.TCombobox", foreground="#111827", fieldbackground="#ffffff", background="#ffffff", selectforeground="#111827", selectbackground="#dbeafe")
        self.access_page = tk.Frame(self, bg="#101820", padx=32, pady=32)
        self.access_page.pack(fill="both", expand=True)
        card = tk.Frame(self.access_page, bg="#17232e", padx=36, pady=32)
        card.place(relx=0.5, rely=0.5, anchor="center", width=520, height=390)
        tk.Label(card, text="FaceTrack Live", font=("Segoe UI", 24, "bold"), bg="#17232e", fg="white").pack(pady=(0, 8))
        tk.Label(card, text="Enter the camera access code to continue", font=("Segoe UI", 11), bg="#17232e", fg="#cbd5e1").pack(pady=(0, 28))
        tk.Label(card, text="Camera access code", anchor="w", bg="#17232e", fg="#dbeafe").pack(fill="x")
        ttk.Entry(card, textvariable=self.access_code, show="•", font=("Segoe UI", 12)).pack(fill="x", pady=(6, 22), ipady=4)
        buttons = tk.Frame(card, bg="#17232e")
        buttons.pack(fill="x")
        self.continue_button = ttk.Button(buttons, text="Continue to camera", command=self._open_camera_page)
        self.continue_button.pack(side="right")
        self.access_status = tk.StringVar(value="The code determines which college is used.")
        tk.Label(card, textvariable=self.access_status, wraplength=440, justify="left", bg="#17232e", fg="#94a3b8").pack(fill="x", pady=(24, 0))

    def _open_camera_page(self) -> None:
        if not self.access_code.get().strip():
            messagebox.showerror("Access code required", "Enter the camera access code for this college.")
            return
        self.continue_button.configure(state="disabled")
        self.access_status.set("Checking access code and loading college data…")
        self.update_idletasks()
        try:
            self._load_college_data()
        except Exception as error:
            self.continue_button.configure(state="normal")
            self.access_status.set(str(error))
            return
        self.access_page.destroy()
        self._build_ui()
        self._apply_recognition_settings(self.settings)
        self.status.set(f"Loaded {len(self.students)} registered face(s) for {self.selected_college_slug}. Start the camera when ready.")

    def _load_college_data(self) -> None:
        self.repository = AttendanceRepository(self.access_code.get())
        self.selected_college_slug = self.repository.college_slug
        self.settings = self.repository.recognition_settings()
        self.students = self.repository.students_with_faces()
        if not self.students:
            raise RuntimeError("No registered face encodings were found for this college.")

    def _build_ui(self) -> None:
        controls = tk.Frame(self, bg="#17232e", padx=16, pady=14)
        controls.pack(fill="x")
        ttk.Style(self).theme_use("clam")
        for label, variable, values, width in (("Camera", self.camera_index, (0, 1, 2, 3), 6), ("Quality", self.resolution, tuple(RESOLUTIONS), 8), ("FPS", self.target_fps, ("15", "24", "30", "60"), 6)):
            tk.Label(controls, text=label, bg="#17232e", fg="#dbeafe").pack(side="left", padx=(0, 5))
            ttk.Combobox(controls, textvariable=variable, values=values, width=width, state="readonly").pack(side="left", padx=(0, 14))
        tk.Label(controls, text="Recognition threshold", bg="#17232e", fg="#dbeafe").pack(side="left", padx=(0, 6))
        self.threshold_label = tk.Label(controls, text="Dashboard setting", bg="#17232e", fg="#dbeafe", width=22)
        self.threshold_label.pack(side="left", padx=(5, 14))
        self.sound_label = tk.Label(controls, text="Sound: dashboard setting", bg="#17232e", fg="#dbeafe")
        self.sound_label.pack(side="left", padx=(0, 14))
        ttk.Button(controls, text="Start camera", command=self.start).pack(side="left", padx=4)
        ttk.Button(controls, text="Stop", command=self.stop).pack(side="left", padx=4)
        body = tk.Frame(self, bg="#101820", padx=16, pady=16)
        body.pack(fill="both", expand=True)
        self.video_label = tk.Label(body, text="Camera preview", bg="#050a0f", fg="#9ca3af", font=("Segoe UI", 18), anchor="center")
        self.video_label.pack(side="left", fill="both", expand=True)
        info = tk.Frame(body, bg="#17232e", width=280, padx=18, pady=20)
        info.pack(side="right", fill="y", padx=(16, 0))
        info.pack_propagate(False)
        tk.Label(info, text="LIVE ATTENDANCE", font=("Segoe UI", 12, "bold"), bg="#17232e", fg="#60a5fa").pack(anchor="w")
        tk.Label(info, textvariable=self.person, justify="left", wraplength=240, font=("Segoe UI", 16, "bold"), bg="#17232e", fg="white").pack(anchor="w", pady=(22, 14))
        tk.Label(info, textvariable=self.liveness_status, justify="left", wraplength=240, font=("Segoe UI", 11, "bold"), bg="#17232e", fg="#fbbf24").pack(anchor="w", pady=(0, 20))
        tk.Label(info, text="Security rules", font=("Segoe UI", 11, "bold"), bg="#17232e", fg="white").pack(anchor="w")
        tk.Label(info, text="• One face only\n• AI anti-spoofing must pass\n• Blink challenge must pass\n• Registered student required\n• Attendance once per day", justify="left", bg="#17232e", fg="#cbd5e1").pack(anchor="w", pady=(8, 22))
        tk.Label(self, textvariable=self.status, anchor="w", padx=16, pady=10, bg="#0b1118", fg="#cbd5e1").pack(fill="x")

    def start(self) -> None:
        self.stop()
        try:
            self._load_college_data()
            self._apply_recognition_settings(self.settings)
            self.camera = cv2.VideoCapture(self.camera_index.get(), cv2.CAP_DSHOW)
            width, height = RESOLUTIONS[self.resolution.get()]
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, int(self.target_fps.get()))
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.camera.isOpened():
                raise RuntimeError("Camera could not be opened. Check the selected camera number.")
            self.running = True
            self.scan_token += 1
            self.capture_stop.clear()
            self.liveness.reset()
            self.latest_recognition = None
            self.next_recognition_at = 0.0
            self.liveness_status.set("Liveness: AI check + blink required")
            self.capture_thread = Thread(target=self._capture_loop, args=(self.camera,), daemon=True)
            self.capture_thread.start()
            self.status.set(f"Scanning {self.selected_college_slug}. A real face must pass AI anti-spoofing and blink once before attendance.")
            self._next_frame()
        except Exception as error:
            self.stop()
            messagebox.showerror("Unable to start", str(error))

    def stop(self) -> None:
        self.running = False
        self.scan_token += 1
        self.capture_stop.set()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=0.5)
        self.capture_thread = None
        with self.frame_lock:
            self.latest_frame = None
        self.latest_recognition = None
        if hasattr(self, "liveness_status"):
            self.liveness_status.set("Liveness: waiting")
        if hasattr(self, "status"):
            self.status.set("Camera stopped.")

    def _capture_loop(self, camera: cv2.VideoCapture) -> None:
        while not self.capture_stop.is_set() and camera.isOpened():
            ok, frame = camera.read()
            if not ok:
                continue
            with self.frame_lock:
                self.latest_frame = frame

    def _next_frame(self) -> None:
        if not self.running or self.camera is None:
            return
        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is None:
            self.after(15, self._next_frame)
            return
        self._collect_recognition()
        self._draw_latest_recognition(frame)

        # Keep preview work cheap: resize before converting to PIL instead of
        # converting the full 720p/1080p frame on every UI tick.
        preview_frame = frame
        max_width, max_height = 850, 640
        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)
        if scale < 1.0:
            preview_frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        preview = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=preview, text="")
        self.video_label.image = preview
        self._schedule_recognition(frame)
        self.after(max(1, round(1000 / int(self.target_fps.get()))), self._next_frame)

    def _schedule_recognition(self, frame: np.ndarray) -> None:
        if self.recognition_future is not None or time.monotonic() < self.next_recognition_at:
            return
        # Face recognition + landmarks are expensive. The camera itself remains
        # at the selected FPS while recognition samples only twice per second.
        self.next_recognition_at = time.monotonic() + 0.50
        token = self.scan_token
        students = self.students
        threshold = self.threshold.get()
        self.recognition_future = self.recognizer.submit(self._recognize, frame, students, threshold)
        self.recognition_future.token = token  # type: ignore[attr-defined]

    def _collect_recognition(self) -> None:
        if self.recognition_future is None or not self.recognition_future.done():
            return
        future, self.recognition_future = self.recognition_future, None
        if future.token != self.scan_token:  # type: ignore[attr-defined]
            return
        try:
            self.latest_recognition = future.result()
        except Exception as error:
            self.status.set(f"Recognition error: {error}")
            return
        self._update_recognition_status()

    def _recognize(self, frame: np.ndarray, students: list[RegisteredStudent], threshold: float) -> RecognitionResult:
        # Detect and encode at 1/4 resolution; this is the existing recognition path.
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_small, model="hog")
        if len(locations) != 1:
            self.liveness.reset()
            return locations, None, None, False, "No face detected" if not locations else "Only one person may be in frame"

        full_location = tuple(value * 4 for value in locations[0])
        live = self.liveness.evaluate(frame, full_location)
        if not live.allowed:
            return locations, None, None, False, live.message

        encoding = face_recognition.face_encodings(rgb_small, locations)[0]
        distances = face_recognition.face_distance([s.encoding for s in students], encoding)
        match_index = int(np.argmin(distances))
        distance = float(distances[match_index])
        return locations, match_index if distance <= threshold else None, distance, True, live.message

    def _draw_latest_recognition(self, frame: np.ndarray) -> None:
        if self.latest_recognition is None:
            return
        locations, match_index, distance, live_ok, message = self.latest_recognition
        if len(locations) != 1:
            for top, right, bottom, left in locations:
                cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 165, 255), 3)
            return
        top, right, bottom, left = locations[0]
        box = (left * 4, top * 4, right * 4, bottom * 4)
        if not live_ok:
            self._draw_box(frame, box, "LIVENESS REQUIRED", (0, 165, 255))
            return
        if match_index is None:
            self._draw_box(frame, box, "UNKNOWN", (0, 0, 255))
            return
        student = self.students[match_index]
        self._draw_box(frame, box, f"{student.name} | {student.roll_no}", (34, 197, 94))

    def _update_recognition_status(self) -> None:
        if self.latest_recognition is None:
            return
        locations, match_index, distance, live_ok, message = self.latest_recognition
        self.liveness_status.set(f"Liveness: {'VERIFIED' if live_ok else message}")
        if len(locations) != 1:
            text = "No face detected" if not locations else "Only one person may be in frame"
            self.person.set(text)
            self.status.set(text)
            return
        if not live_ok:
            self.person.set("Face detected\nAttendance locked")
            self.status.set(message)
            return
        if match_index is None:
            self.person.set(f"Unknown face\nDistance: {distance:.3f}")
            self.status.set("Live face verified, but the face is not registered or is below the selected confidence.")
            return
        student = self.students[match_index]
        self.person.set(f"{student.name}\n{student.roll_no}\n{student.department}\nMatch distance: {distance:.3f}")
        now = time.monotonic()
        if student.id != self.last_match_id or now - self.last_match_at > 5:
            self.last_match_id, self.last_match_at = student.id, now
            try:
                created = self.repository.mark_present(student.id) if self.repository else False
                result = "Attendance marked" if created else "Attendance already marked today"
                self.status.set(f"{result}: {student.name}")
                if self.sound_enabled.get():
                    self._play_recognition_sound()
            except Exception as error:
                self.status.set(f"Recognized {student.name}, but attendance failed: {error}")

    @staticmethod
    def _draw_box(frame: np.ndarray, box: tuple[int, int, int, int], label: str, colour: tuple[int, int, int]) -> None:
        left, top, right, bottom = box
        cv2.rectangle(frame, (left, top), (right, bottom), colour, 3)
        cv2.rectangle(frame, (left, max(0, top - 30)), (right, top), colour, cv2.FILLED)
        cv2.putText(frame, label, (left + 6, top - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    def _apply_recognition_settings(self, settings: RecognitionSettings) -> None:
        self.threshold.set(settings.distance_threshold)
        self.sound_enabled.set(settings.sound_alerts)
        self.threshold_label.configure(text=f"{settings.confidence_threshold}% (distance {settings.distance_threshold:.2f})")
        self.sound_label.configure(text=f"Sound: {'on' if settings.sound_alerts else 'off'} (dashboard)")

    @staticmethod
    def _play_recognition_sound() -> None:
        try:
            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except RuntimeError:
            winsound.MessageBeep(winsound.MB_OK)

    def close(self) -> None:
        self.stop()
        self.recognizer.shutdown(wait=False, cancel_futures=True)
        self.destroy()


if __name__ == "__main__":
    FaceAttendanceApp().mainloop()
