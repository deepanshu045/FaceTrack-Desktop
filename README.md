# FaceTrack Live Attendance

A Windows desktop camera app for the existing `D:\College_Project` backend. It permits exactly one face in frame, matches that face against registered student encodings, and records daily attendance in the backend database.

## Run

Use the existing backend virtual environment, which already includes `face_recognition` and its native dependencies:

```powershell
cd D:\faceRecognitionSoftware
D:\College_Project\myenv\Scripts\python.exe -m pip install -r requirements.txt
D:\College_Project\myenv\Scripts\python.exe app.py
```

The app defaults to `D:\College_Project`. If the backend is elsewhere, set `FACE_ATTENDANCE_BACKEND` to its directory before starting. The backend `.env` must contain its normal database settings.

Controls: choose the camera number, quality, and target FPS before **Start camera**. The desktop app reads the authenticated college's dashboard confidence threshold and sound-alert setting each time scanning starts. The dashboard's 60–99% confidence range maps to a face-distance limit of 0.70–0.50 (higher confidence is stricter). Windows plays the system recognition sound for both newly recorded and already-recorded attendance when sound alerts are enabled.
