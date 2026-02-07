"""Video to Sprite Sheet — Upload, frame range selection."""

import io
import shutil
import uuid
import zipfile
from pathlib import Path

import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm"}

app = FastAPI(title="Video to Sprite Sheet", version="0.1.0")

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _check_extension(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page."""
    html_path = PROJECT_ROOT / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="Frontend not found")
    return FileResponse(html_path)


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file. Returns the URL to stream the uploaded file for preview.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not _check_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save with unique name to avoid collisions
    ext = Path(file.filename).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = UPLOAD_DIR / unique_name

    try:
        with file_path.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    return {
        "filename": file.filename,
        "id": unique_name,
        "url": f"/api/video/{unique_name}",
    }


@app.get("/api/video/{filename}")
async def stream_video(filename: str):
    """Stream the uploaded video for preview."""
    # Security: allow only filenames we generated (hex + ext)
    name = Path(filename).name
    if not name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = UPLOAD_DIR / name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    mime_map = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".webm": "video/webm",
    }
    ext = Path(name).suffix.lower()
    media_type = mime_map.get(ext, "video/*")

    return FileResponse(file_path, media_type=media_type)


def _get_video_path(filename: str) -> Path:
    """Validate filename and return full path to video."""
    name = Path(filename).name
    if not name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = UPLOAD_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return path


@app.get("/api/video/{filename}/metadata")
async def get_video_metadata(filename: str):
    """Return frame count, fps, and dimensions for frame range selection."""
    path = _get_video_path(filename)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return {
            "frame_count": max(0, frame_count),
            "fps": round(fps, 2),
            "width": width,
            "height": height,
        }
    finally:
        cap.release()


@app.get("/api/video/{filename}/frames")
async def get_all_frames(filename: str):
    """Extract all frames and return as a ZIP file for frontend caching."""
    path = _get_video_path(filename)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video")
    zip_buf = io.BytesIO()
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = max(0, frame_count)
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                zf.writestr(f"frame_{i:05d}.png", buf.getvalue())
    finally:
        cap.release()
    zip_buf.seek(0)
    return Response(
        content=zip_buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=frames.zip"},
    )


@app.get("/api/video/{filename}/frame")
async def get_frame(
    filename: str,
    index: int = Query(..., ge=0, description="Frame index"),
):
    """Extract and return a single frame as PNG image."""
    path = _get_video_path(filename)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise HTTPException(status_code=404, detail="Frame not found")
        # OpenCV uses BGR, convert to RGB for correct display in browser
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    finally:
        cap.release()


# Mount static assets (CSS, JS) if present
static_dir = PROJECT_ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
