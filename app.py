"""FaceTrack Live - live face attendance with anti-spoofing and liveness."""

from __future__ import annotations

import ctypes
import sys
import time

if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            pass

import tkinter as tk
import winsound
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock, Thread
from tkinter import messagebox, ttk

import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageTk

from backend_bridge import ActiveLecture, AttendanceRepository, RecognitionSettings, RegisteredStudent
from liveness import LivenessGuard

RESOLUTIONS = {"480p": (640, 480), "720p": (1280, 720), "1080p": (1920, 1080)}
RECOGNITION_SCALE = 0.25
# Keep the preview smooth while doing expensive HOG work less often.
RECOGNITION_INTERVAL = 0.65
PREVIEW_INTERVAL_MS = 33
RecognitionResult = tuple[list[tuple[int, int, int, int]], int | None, float | None, bool, str]

class FaceAttendanceApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FaceTrack Live Attendance")
        self.geometry("1360x820")
        self.minsize(1120, 720)
        self.configure(bg="#070d18")
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
        self.known_encodings: np.ndarray = np.empty((0, 128), dtype=np.float64)
        self.active_lecture: ActiveLecture | None = None
        self.last_match_id: int | None = None
        self.last_match_at = 0.0
        self.running = False
        self.liveness = LivenessGuard()
        self.camera_index = tk.IntVar(value=0)
        self.camera_choice = tk.StringVar(value="")
        self.camera_options: list[str] = []
        self.camera_devices: dict[str, int] = {}
        self.resolution = tk.StringVar(value="480p")
        self.target_fps = tk.StringVar(value="30")
        self.threshold = tk.DoubleVar(value=0.50)
        self.sound_enabled = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Ready to scan.")
        self.person = tk.StringVar(value="No active scan")
        self.liveness_status = tk.StringVar(value="Liveness waiting")
        self.lecture_status = tk.StringVar(value="Waiting for student")
        self.access_code = tk.StringVar()
        self.selected_college_slug = ""
        self._configure_styles()
        self._build_college_access_page()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Dark.TCombobox", foreground="#e6edf7", fieldbackground="#0e1727", background="#0e1727", bordercolor="#2a3950", lightcolor="#0e1727", darkcolor="#0e1727", selectforeground="#ffffff", selectbackground="#2455a6", arrowcolor="#9fb1c8")
        style.configure("Blue.TButton", foreground="#ffffff", background="#2457c5", borderwidth=0, padding=(15, 9), font=("Segoe UI", 10, "bold"))
        style.map("Blue.TButton", background=[("active", "#2e6ce6"), ("disabled", "#233a68")])
        style.configure("Secondary.TButton", foreground="#dbe7f5", background="#263449", borderwidth=0, padding=(13, 9), font=("Segoe UI", 10, "bold"))
        style.map("Secondary.TButton", background=[("active", "#31425b")])

    @staticmethod
    def _label(parent, text="", size=10, weight="normal", fg="#dbe7f5", bg="#172234", **kwargs):
        return tk.Label(parent, text=text, font=("Segoe UI", size, weight), fg=fg, bg=bg, **kwargs)

    @staticmethod
    def _card(parent, bg="#172234", padx=18, pady=16):
        return tk.Frame(parent, bg=bg, highlightbackground="#27364c", highlightthickness=1, padx=padx, pady=pady)

    def _build_college_access_page(self) -> None:
        page = tk.Frame(self, bg="#070d18")
        page.pack(fill="both", expand=True)
        self.access_page = page
        top = tk.Frame(page, bg="#070d18", padx=34, pady=24)
        top.pack(fill="x")
        self._label(top, "FaceTrack", 22, "bold", "#f5f8fd", "#070d18").pack(side="left")
        self._label(top, "LIVE ATTENDANCE", 9, "bold", "#6ea8ff", "#070d18").pack(side="left", padx=12)
        card = self._card(page, bg="#172234", padx=42, pady=34)
        card.place(relx=0.5, rely=0.50, anchor="center", width=520, height=380)
        self._label(card, "Camera access", 25, "bold", "#f5f8fd").pack(anchor="w")
        self._label(card, "Connect this scanner to the correct college workspace.", 10, fg="#91a1b8").pack(anchor="w", pady=(6, 28))
        self._label(card, "ACCESS CODE", 9, "bold", "#7f94b1").pack(anchor="w")
        entry = tk.Entry(card, textvariable=self.access_code, show="•", bg="#0e1727", fg="#eef5ff", insertbackground="#eef5ff", relief="flat", font=("Segoe UI", 13), bd=0)
        entry.pack(fill="x", ipady=11, pady=(7, 18))
        entry.bind("<Return>", lambda _event: self._open_camera_page())
        self.continue_button = ttk.Button(card, text="Continue to scanner", style="Blue.TButton", command=self._open_camera_page)
        self.continue_button.pack(fill="x")
        self.access_status = tk.StringVar(value="The access code determines which college is used.")
        tk.Label(card, textvariable=self.access_status, font=("Segoe UI", 9), fg="#8292aa", bg="#172234", wraplength=420, justify="left").pack(anchor="w", pady=(16, 0))

    def _open_camera_page(self) -> None:
        if not self.access_code.get().strip():
            messagebox.showerror("Access code required", "Enter the camera access code for this college.")
            return
        self.continue_button.configure(state="disabled")
        self.access_status.set("Checking access code and loading face profiles…")
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
        self._refresh_active_lecture()
        self._refresh_cameras()
        self.status.set(f"{len(self.students)} registered face profiles ready.")

    def _load_college_data(self) -> None:
        old_repository = self.repository
        if old_repository is not None:
            old_repository.stop_heartbeat()
        self.repository = AttendanceRepository(self.access_code.get())
        self.selected_college_slug = self.repository.college_slug
        self.settings = self.repository.recognition_settings()
        self.students = self.repository.students_with_faces()
        self.known_encodings = self._build_known_encodings(self.students)
        self.active_lecture = None
        if not self.students:
            raise RuntimeError("No registered face encodings were found for this college.")

    @staticmethod
    def _build_known_encodings(students: list[RegisteredStudent]) -> np.ndarray:
        if not students:
            return np.empty((0, 128), dtype=np.float64)
        return np.ascontiguousarray(np.asarray([s.encoding for s in students], dtype=np.float64))

    def _refresh_active_lecture(self, student_id: int | None = None) -> None:
        if self.repository is None:
            self.active_lecture = None
            self.lecture_status.set("No college selected")
            return
        if student_id is None:
            self.active_lecture = None
            self.lecture_status.set("Waiting for student")
            if hasattr(self, "lecture_stat"):
                self.lecture_stat.configure(text="Waiting for student")
            return
        try:
            self.active_lecture = self.repository.active_lecture_for_student(student_id)
        except Exception as error:
            self.active_lecture = None
            self.lecture_status.set(f"Lecture check failed: {error}")
            return
        if self.active_lecture is None:
            self.lecture_status.set("No active lecture for this student")
            return
        lecture = self.active_lecture
        class_name = f"{lecture.class_name} {lecture.section}".strip()
        self.lecture_status.set(f"{lecture.subject}  •  {class_name}  •  {lecture.start_time}–{lecture.end_time}")

    def _build_ui(self) -> None:
        header = tk.Frame(self, bg="#0b1321", padx=22, pady=12)
        header.pack(fill="x")
        self._label(header, "FaceTrack", 20, "bold", "#f4f7fb", "#0b1321").pack(side="left")
        self._label(header, "LIVE ATTENDANCE", 9, "bold", "#70a8ff", "#0b1321").pack(side="left", padx=(10, 0))
        self.change_access_button = ttk.Button(header, text="Change access code", style="Secondary.TButton", command=self._change_access_code)
        self.change_access_button.pack(side="right", padx=(14, 0))
        self.header_status = self._label(header, "● Scanner offline", 9, "bold", "#8c9bb0", "#0b1321")
        self.header_status.pack(side="right")
        stats = tk.Frame(self, bg="#070d18", padx=16, pady=12)
        stats.pack(fill="x")
        self._stat_card(stats, "SCANNER STATUS", "Ready to start", "◉", "scanner_stat")
        self._stat_card(stats, "CURRENT LECTURE", "Waiting for student", "▣", "lecture_stat")
        self._stat_card(stats, "FACE PROFILES READY", str(len(self.students)), "♙", "profiles_stat")
        content = tk.Frame(self, bg="#070d18", padx=16)
        content.pack(fill="both", expand=True, pady=(0, 12))
        content.grid_columnconfigure(0, weight=1, minsize=650)
        content.grid_columnconfigure(1, weight=0, minsize=330)
        content.grid_rowconfigure(0, weight=1)
        left = self._card(content, bg="#172234", padx=16, pady=14)
        left.grid(row=0, column=0, sticky="nsew")
        title_row = tk.Frame(left, bg="#172234")
        title_row.pack(fill="x")
        title = tk.Frame(title_row, bg="#172234")
        title.pack(side="left", fill="x", expand=True)
        self._label(title, "Camera Scanner", 16, "bold").pack(anchor="w")
        self._label(title, "Keep exactly one registered face inside the guide.", 9, fg="#8ea0b9").pack(anchor="w", pady=(3, 0))
        self.start_button = ttk.Button(title_row, text="▣  Start Camera", style="Blue.TButton", command=self.start)
        self.start_button.pack(side="right")
        camera_shell = tk.Frame(left, bg="#03070d", highlightbackground="#1d2b40", highlightthickness=1)
        camera_shell.pack(fill="both", expand=True, pady=(14, 10))
        self.video_label = tk.Label(camera_shell, text="Camera is stopped", bg="#03070d", fg="#73849d", font=("Segoe UI", 12), anchor="center")
        self.video_label.pack(fill="both", expand=True)
        control = tk.Frame(left, bg="#172234")
        control.pack(fill="x")
        self._label(control, "CAMERA", 8, "bold", "#7589a5").pack(side="left")
        self.camera_combo = ttk.Combobox(control, textvariable=self.camera_choice, values=(), width=24, state="readonly", style="Dark.TCombobox")
        self.camera_combo.pack(side="left", padx=(7, 8))
        self.camera_combo.bind("<<ComboboxSelected>>", self._on_camera_selected)
        self.refresh_camera_button = ttk.Button(control, text="Refresh", style="Secondary.TButton", command=self._refresh_cameras)
        self.refresh_camera_button.pack(side="left", padx=(0, 14))
        self._label(control, "QUALITY", 8, "bold", "#7589a5").pack(side="left")
        ttk.Combobox(control, textvariable=self.resolution, values=tuple(RESOLUTIONS), width=7, state="readonly", style="Dark.TCombobox").pack(side="left", padx=(7, 14))
        self._label(control, "FPS", 8, "bold", "#7589a5").pack(side="left")
        ttk.Combobox(control, textvariable=self.target_fps, values=("15", "24", "30", "60"), width=5, state="readonly", style="Dark.TCombobox").pack(side="left", padx=(7, 0))
        ttk.Button(control, text="Stop", style="Secondary.TButton", command=self.stop).pack(side="right")
        right = self._card(content, bg="#172234", padx=14, pady=14)
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        self._label(right, "Live Verification", 15, "bold").pack(anchor="w")
        self._label(right, "Recognition only succeeds after liveness.", 9, fg="#8ea0b9").pack(anchor="w", pady=(2, 10))
        self._verification_pill(right, "CURRENT LECTURE", self.lecture_status, wraplength=270)
        self._verification_pill(right, "LIVENESS", self.liveness_status, wraplength=270)
        self._verification_pill(right, "IDENTITY", self.person, wraplength=270)
        rules = self._card(right, bg="#0e1727", padx=12, pady=10)
        rules.pack(fill="x", pady=(8, 8))
        self._label(rules, "SECURITY CHECKS", 8, "bold", "#7589a5", "#0e1727").pack(anchor="w")
        checks = ("One face only", "AI anti-spoofing", "Blink challenge", "Registered student", "Active lecture only")
        checks_grid = tk.Frame(rules, bg="#0e1727")
        checks_grid.pack(fill="x", pady=(5, 0))
        for index, text in enumerate(checks):
            row = index % 3
            col = index // 3
            self._label(checks_grid, "•  " + text, 8, fg="#c7d4e5", bg="#0e1727").grid(row=row, column=col, sticky="w", padx=(0, 10), pady=2)
        checks_grid.grid_columnconfigure(0, weight=1)
        checks_grid.grid_columnconfigure(1, weight=1)
        settings = self._card(right, bg="#0e1727", padx=12, pady=10)
        settings.pack(fill="x")
        self._label(settings, "SCANNER SETTINGS", 8, "bold", "#7589a5", "#0e1727").pack(anchor="w")
        self.threshold_label = self._label(settings, "Recognition: --", 8, fg="#a7b6ca", bg="#0e1727")
        self.threshold_label.pack(anchor="w", pady=(5, 0))
        self.sound_label = self._label(settings, "Sound alerts: on", 8, fg="#a7b6ca", bg="#0e1727")
        self.sound_label.pack(anchor="w", pady=(2, 0))
        footer = tk.Frame(self, bg="#0b1321", padx=18, pady=8)
        footer.pack(fill="x")
        self._label(footer, "STATUS", 8, "bold", "#7589a5", "#0b1321").pack(side="left", padx=(0, 10))
        self._label(footer, textvariable=self.status, 9, fg="#b9c7d9", bg="#0b1321", anchor="w").pack(side="left", fill="x", expand=True)

    def _stat_card(self, parent, title: str, value: str, icon: str, attr: str) -> None:
        card = tk.Frame(parent, bg="#172234", padx=14, pady=10, highlightbackground="#27364c", highlightthickness=1)
        card.pack(side="left", fill="x", expand=True, padx=4)
        self._label(card, title, 8, "bold", "#7589a5").pack(anchor="w")
        label = self._label(card, value, 10, "bold", "#e8f0fa")
        label.pack(anchor="w", pady=(3, 0))
        setattr(self, attr, label)

    def _verification_pill(self, parent, title: str, variable: tk.StringVar, wraplength: int = 270) -> None:
        frame = tk.Frame(parent, bg="#0e1727", padx=12, pady=9)
        frame.pack(fill="x", pady=4)
        self._label(frame, title, 8, "bold", "#7589a5", "#0e1727").pack(anchor="w")
        self._label(frame, textvariable=variable, 9, "bold", "#dce8f7", "#0e1727", wraplength=wraplength, justify="left").pack(anchor="w", pady=(4, 0))

    def _refresh_cameras(self) -> None:
        if not hasattr(self, "camera_combo"):
            return
        previous = self.camera_choice.get()
        options: list[str] = []
        devices: dict[str, int] = {}
        for index in range(8):
            camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            try:
                if camera.isOpened():
                    ok, _ = camera.read()
                    if ok:
                        name = f"Camera {index}"
                        options.append(name)
                        devices[name] = index
            finally:
                camera.release()
        self.camera_options = options
        self.camera_devices = devices
        self.camera_combo["values"] = options
        if previous in devices:
            self.camera_choice.set(previous)
        elif options:
            self.camera_choice.set(options[0])
            self.camera_index.set(devices[options[0]])
        else:
            self.camera_choice.set("")
            self.status.set("No usable camera detected.")

    def _on_camera_selected(self, _event=None) -> None:
        selected = self.camera_choice.get()
        if selected not in self.camera_devices:
            return
        self.camera_index.set(self.camera_devices[selected])
        if self.running:
            self._restart_selected_camera()
        else:
            self.status.set(f"Selected {selected}. Press Start Camera when ready.")

    def _restart_selected_camera(self) -> None:
        if not self.running:
            return
        self.stop()
        self.start()

    def _change_access_code(self) -> None:
        self.stop()
        if self.repository is not None:
            self.repository.stop_heartbeat()
        self.repository = None
        self.access_code.set("")
        self._build_college_access_page()

    def start(self) -> None:
        if self.running:
            return
        try:
            self._apply_recognition_settings(self.settings)
            selected_index = self.camera_index.get()
            if self.camera_choice.get() in self.camera_devices:
                selected_index = self.camera_devices[self.camera_choice.get()]
            self.camera_index.set(selected_index)
            self.camera = cv2.VideoCapture(selected_index, cv2.CAP_DSHOW)
            width, height = RESOLUTIONS[self.resolution.get()]
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, int(self.target_fps.get()))
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.camera.isOpened():
                raise RuntimeError("Selected camera could not be opened. Click Refresh and choose another camera.")
            self.running = True
            self.scan_token += 1
            self.capture_stop.clear()
            self.liveness.reset()
            self.latest_recognition = None
            self.next_recognition_at = 0.0
            self.active_lecture = None
            self.lecture_status.set("Waiting for student")
            self.lecture_stat.configure(text="Waiting for student")
            self.liveness_status.set("Live face check + blink required")
            self.person.set("Face not verified")
            self.header_status.configure(text="● Scanner active", fg="#22c995")
            self.scanner_stat.configure(text="FaceTrack scanning")
            self.start_button.configure(text="▣  Camera Active")
            self.capture_thread = Thread(target=self._capture_loop, args=(self.camera,), daemon=True)
            self.capture_thread.start()
            self.status.set(f"Scanning with Camera {selected_index}. Keep exactly one face inside the guide and blink once.")
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
            self.liveness_status.set("Waiting for camera")
        if hasattr(self, "person"):
            self.person.set("No active scan")
        if hasattr(self, "lecture_status"):
            self.lecture_status.set("Waiting for student")
        if hasattr(self, "header_status"):
            self.header_status.configure(text="● Scanner offline", fg="#8c9bb0")
        if hasattr(self, "scanner_stat"):
            self.scanner_stat.configure(text="Ready to start")
        if hasattr(self, "lecture_stat"):
            self.lecture_stat.configure(text="Waiting for student")
        if hasattr(self, "start_button"):
            self.start_button.configure(text="▣  Start Camera")
        if hasattr(self, "video_label"):
            self.video_label.configure(image="", text="Camera is stopped")
            self.video_label.image = None
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
        label_width = max(self.video_label.winfo_width() - 8, 320)
        label_height = max(self.video_label.winfo_height() - 8, 240)
        height, width = frame.shape[:2]
        scale = min(label_width / width, label_height / height)
        preview_frame = frame
        if scale != 1.0:
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            preview_frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=interpolation)
        rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        preview = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=preview, text="")
        self.video_label.image = preview
        self._schedule_recognition(frame)
        self.after(PREVIEW_INTERVAL_MS, self._next_frame)

    def _schedule_recognition(self, frame: np.ndarray) -> None:
        if self.recognition_future is not None or time.monotonic() < self.next_recognition_at:
            return
        self.next_recognition_at = time.monotonic() + RECOGNITION_INTERVAL
        token = self.scan_token
        students = self.students
        known_encodings = self.known_encodings
        threshold = self.threshold.get()
        self.recognition_future = self.recognizer.submit(self._recognize, frame, students, known_encodings, threshold)
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

    def _recognize(self, frame: np.ndarray, students: list[RegisteredStudent], known_encodings: np.ndarray, threshold: float) -> RecognitionResult:
        small = cv2.resize(frame, (0, 0), fx=RECOGNITION_SCALE, fy=RECOGNITION_SCALE, interpolation=cv2.INTER_AREA)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_small, number_of_times_to_upsample=0, model="hog")
        if len(locations) != 1:
            if not locations:
                missing = self.liveness.evaluate_missing_face(frame)
                recent = self.liveness.recent_face_location
                if recent is not None:
                    cached = tuple(max(0, int(value / RECOGNITION_SCALE)) for value in recent)
                    return [cached], None, None, False, missing.message
                return [], None, None, False, missing.message
            self.liveness.reset()
            return locations, None, None, False, "Only one person may be in frame"
        full_location = tuple(int(value / RECOGNITION_SCALE) for value in locations[0])
        live = self.liveness.evaluate(frame, full_location)
        if not live.allowed:
            return locations, None, None, False, live.message
        encoding = face_recognition.face_encodings(rgb_small, known_face_locations=locations, num_jitters=1, model="small")[0]
        if known_encodings.size == 0:
            return locations, None, None, True, live.message
        # Avoid the extra sqrt performed by face_distance; compare squared distances instead.
        delta = known_encodings - encoding
        squared_distances = np.einsum("ij,ij->i", delta, delta)
        match_index = int(np.argmin(squared_distances))
        squared_distance = float(squared_distances[match_index])
        distance = squared_distance ** 0.5
        return locations, match_index if squared_distance <= threshold * threshold else None, distance, True, live.message

    def _draw_latest_recognition(self, frame: np.ndarray) -> None:
        if self.latest_recognition is None:
            h, w = frame.shape[:2]
            cv2.ellipse(frame, (w // 2, h // 2), (max(90, w // 8), max(120, h // 4)), 0, 0, 360, (80, 130, 210), 2)
            return
        locations, match_index, distance, live_ok, _message = self.latest_recognition
        if len(locations) != 1:
            return
        top, right, bottom, left = locations[0]
        box = (int(left / RECOGNITION_SCALE), int(top / RECOGNITION_SCALE), int(right / RECOGNITION_SCALE), int(bottom / RECOGNITION_SCALE))
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
        self.liveness_status.set("Verified ✓" if live_ok else message)
        if len(locations) != 1:
            text = "No face detected" if not locations else "Only one person may be in frame"
            self.person.set(text)
            self.status.set(text)
            return
        if not live_ok:
            self.person.set("Face detected • Attendance locked")
            self.status.set(message)
            return
        if match_index is None:
            if distance is None:
                self.person.set("Blink verified • Face reacquiring…")
                self.status.set(message)
                return
            self.person.set(f"Unknown face • distance {distance:.3f}")
            self.status.set("Live face verified, but the face is not registered.")
            return
        student = self.students[match_index]
        self.person.set(f"{student.name} • {student.roll_no}")
        self.status.set(f"Live face verified: {student.name}")
        self.scanner_stat.configure(text="Face verified")
        now = time.monotonic()
        if student.id != self.last_match_id or now - self.last_match_at > 5:
            self.last_match_id, self.last_match_at = student.id, now
            try:
                self._refresh_active_lecture(student.id)
                if self.active_lecture is None:
                    self.lecture_stat.configure(text="Attendance closed")
                    self.status.set(f"Recognized {student.name}, but there is no active lecture for this student.")
                    return
                lecture = self.active_lecture
                self.lecture_stat.configure(text=f"{lecture.subject} • {lecture.start_time}–{lecture.end_time}")
                result = self.repository.mark_present(student.id, lecture.id) if self.repository else {}
                if result.get("already_marked"):
                    self.status.set(f"Already marked for {lecture.subject}: {student.name}")
                else:
                    self.status.set(f"Attendance marked for {lecture.subject}: {student.name}")
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
        self.threshold_label.configure(text=f"Recognition: {settings.confidence_threshold}%  •  distance {settings.distance_threshold:.2f}")
        self.sound_label.configure(text=f"Sound alerts: {'on' if settings.sound_alerts else 'off'}")

    @staticmethod
    def _play_recognition_sound() -> None:
        try:
            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except RuntimeError:
            winsound.MessageBeep(winsound.MB_OK)

    def close(self) -> None:
        self.stop()
        if self.repository is not None:
            self.repository.stop_heartbeat()
        self.recognizer.shutdown(wait=False, cancel_futures=True)
        self.destroy()

if __name__ == "__main__":
    FaceAttendanceApp().mainloop()
