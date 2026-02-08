"""Video to Sprite Sheet — Upload, frame range selection."""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
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


def _rife_exe_path() -> Optional[Path]:
    """Return path to rife-ncnn-vulkan executable, or None if not found."""
    exe_name = "rife-ncnn-vulkan.exe" if sys.platform == "win32" else "rife-ncnn-vulkan"

    # Env override
    env = os.environ.get("RIFE_NCNN_VULKAN") or os.environ.get("RIFE_NCNN_VULKAN_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        if p.is_dir():
            c = p / exe_name
            if c.exists():
                return c

    # Project root: rife-ncnn-vulkan extracted in project root
    candidates = [
        PROJECT_ROOT / "rife-ncnn-vulkan" / exe_name,
        PROJECT_ROOT / "rife-ncnn-vulkan-win64" / exe_name,
        PROJECT_ROOT / exe_name,
    ]
    for c in candidates:
        if c.exists():
            return c

    # Scan for any folder starting with "rife-ncnn-vulkan" (e.g. rife-ncnn-vulkan-20240210-windows-vs2019)
    try:
        for item in PROJECT_ROOT.iterdir():
            if item.is_dir() and item.name.lower().startswith("rife-ncnn-vulkan"):
                c = item / exe_name
                if c.exists():
                    return c
    except OSError:
        pass

    # Fallback: PATH
    which = shutil.which(exe_name)
    return Path(which) if which else None


MERGE_DIST = 6  # merge colors within this distance in RGB space
RGB_CUBE_CORNERS = [
    (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255),
]


def _apply_despill(data: np.ndarray, kr: int, kg: int, kb: int, strength: float) -> None:
    """Remove color spill from foreground pixels (in-place)."""
    s = strength / 100.0
    r = data[:, :, 0].astype(np.float64)
    g = data[:, :, 1].astype(np.float64)
    b = data[:, :, 2].astype(np.float64)
    alpha = data[:, :, 3]
    mask = alpha > 0
    if kg == kb and kg >= kr:
        spill = np.maximum(0, np.minimum(g, b) - r)
        g = np.where(mask, np.clip(np.round(g - spill * s), 0, 255), g)
        b = np.where(mask, np.clip(np.round(b - spill * s), 0, 255), b)
    elif kr == kb and kr >= kg:
        spill = np.maximum(0, np.minimum(r, b) - g)
        r = np.where(mask, np.clip(np.round(r - spill * s), 0, 255), r)
        b = np.where(mask, np.clip(np.round(b - spill * s), 0, 255), b)
    elif kr == kg and kr >= kb:
        spill = np.maximum(0, np.minimum(r, g) - b)
        r = np.where(mask, np.clip(np.round(r - spill * s), 0, 255), r)
        g = np.where(mask, np.clip(np.round(g - spill * s), 0, 255), g)
    elif kg >= kr and kg >= kb:
        spill = np.maximum(0, g - np.maximum(r, b))
        g = np.where(mask, np.clip(np.round(g - spill * s), 0, 255), g)
    elif kb >= kr and kb >= kg:
        spill = np.maximum(0, b - np.maximum(r, g))
        b = np.where(mask, np.clip(np.round(b - spill * s), 0, 255), b)
    elif kr >= kg and kr >= kb:
        spill = np.maximum(0, r - np.maximum(g, b))
        r = np.where(mask, np.clip(np.round(r - spill * s), 0, 255), r)
    data[:, :, 0] = r.astype(np.uint8)
    data[:, :, 1] = g.astype(np.uint8)
    data[:, :, 2] = b.astype(np.uint8)


def _apply_cluster_filter(data: np.ndarray, w: int, h: int, max_clusters: int) -> None:
    """Keep only top max_clusters by size; zero alpha elsewhere (in-place)."""
    if max_clusters <= 0:
        return
    alpha = data[:, :, 3]
    cluster_id = np.full((h, w), -1, dtype=np.int32)
    cluster_id[alpha > 0] = 0
    clusters: List[tuple] = []
    next_id = 1
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    for y in range(h):
        for x in range(w):
            if cluster_id[y, x] != 0:
                continue
            stack = [(x, y)]
            cluster_id[y, x] = next_id
            size = 1
            while stack:
                cx, cy = stack.pop()
                for d in range(8):
                    nx, ny = cx + dx[d], cy + dy[d]
                    if 0 <= nx < w and 0 <= ny < h and cluster_id[ny, nx] == 0:
                        cluster_id[ny, nx] = next_id
                        size += 1
                        stack.append((nx, ny))
            clusters.append((next_id, size))
            next_id += 1
    clusters.sort(key=lambda c: -c[1])
    keep_ids = {c[0] for c in clusters[:max_clusters]}
    for y in range(h):
        for x in range(w):
            cid = cluster_id[y, x]
            if cid > 0 and cid not in keep_ids:
                data[y, x, 3] = 0


def _apply_chroma_removal(
    rgba: np.ndarray,
    key_rgb: tuple,
    tolerance: int,
    holo_enabled: bool,
    holo_strength: int,
    max_clusters: int,
) -> np.ndarray:
    """
    Apply chroma key removal using the same logic as Step 3 (Pick color & remove background).
    Returns RGBA array (may modify in-place).
    """
    kr, kg, kb = key_rgb
    threshold = (tolerance / 100.0) * 442.0
    r, g, b = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2]
    dist = np.sqrt((r.astype(np.float64) - kr) ** 2 + (g.astype(np.float64) - kg) ** 2 + (b.astype(np.float64) - kb) ** 2)
    rgba[:, :, 3] = np.where(dist <= threshold, 0, rgba[:, :, 3])
    # Skip despill for RIFE output: synthetic cyan blend differs from real green-screen spill;
    # despill distorts character colors (e.g. wrench edges) instead of fixing them.
    if max_clusters > 0:
        h, w = rgba.shape[:2]
        _apply_cluster_filter(rgba, w, h, max_clusters)
    return rgba


def _pick_chroma_color(pil1: Image.Image, pil2: Image.Image) -> tuple:
    """
    Find a color maximally distant from image colors.
    1. Collect non-transparent pixels from both images.
    2. Quantize to MERGE_DIST grid to reduce the set.
    3. Pick RGB cube corner with maximum min-distance to reduced set.
    """
    def _pixels_to_reduced_set(pil: Image.Image) -> set:
        arr = np.array(pil.convert("RGBA"))
        r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
        mask = (a >= 128)
        pixels = set(zip(r[mask].tolist(), g[mask].tolist(), b[mask].tolist()))
        return pixels

    raw = _pixels_to_reduced_set(pil1) | _pixels_to_reduced_set(pil2)
    if not raw:
        return (0, 255, 0)

    def _quantize(c: tuple) -> tuple:
        return (c[0] // MERGE_DIST, c[1] // MERGE_DIST, c[2] // MERGE_DIST)

    reduced = set(_quantize(c) for c in raw)
    centers = [(q[0] * MERGE_DIST + MERGE_DIST // 2,
                q[1] * MERGE_DIST + MERGE_DIST // 2,
                q[2] * MERGE_DIST + MERGE_DIST // 2)
               for q in reduced]

    def _dist(a: tuple, b: tuple) -> float:
        return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))

    best_color = RGB_CUBE_CORNERS[0]
    best_min_dist = -1.0
    for cand in RGB_CUBE_CORNERS:
        min_d = min(_dist(cand, c) for c in centers) if centers else 442.0
        if min_d > best_min_dist:
            best_min_dist = min_d
            best_color = cand
    return best_color


def _interpolate_with_rife_cli(
    pil1: Image.Image,
    pil2: Image.Image,
    num_frames: int,
    chroma_tolerance: int = 30,
    chroma_holo_enabled: bool = False,
    chroma_holo_strength: int = 80,
    chroma_max_clusters: int = 0,
) -> Optional[List[bytes]]:
    """
    Dilation + Straight Alpha: extend sprite color into transparent areas.
    Final assembly: full color + alpha mask (no premultiply = no dark edges).
    """
    exe_path = _rife_exe_path()
    if not exe_path or not exe_path.exists():
        print("RIFE: exe not found (checked PROJECT_ROOT=%s)" % PROJECT_ROOT, flush=True)
        return None

    def prepare_frame(pil: Image.Image) -> tuple:
        """
        Dilation: extend sprite color into transparent areas. RIFE gets full bright color.
        """
        arr = np.array(pil.convert("RGBA"))
        bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = arr[:, :, 3]
        mask_holes = (alpha == 0).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        dilated_bgr = bgr.copy()
        for _ in range(5):
            dilated_step = cv2.dilate(dilated_bgr, kernel)
            dilated_bgr = np.where(mask_holes[:, :, np.newaxis] == 255, dilated_step, dilated_bgr)
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        return dilated_bgr, alpha_3ch

    comp1, alpha1 = prepare_frame(pil1)
    comp2, alpha2 = prepare_frame(pil2)
    cwd = exe_path.parent

    def rife_one(path0: Path, path1: Path, out_path: Path) -> bool:
        cmd = [
            str(exe_path),
            "-0", str(path0.resolve()),
            "-1", str(path1.resolve()),
            "-o", str(out_path.resolve()),
        ]
        try:
            proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            print("RIFE: subprocess error: %s" % e, flush=True)
            return False
        if proc.returncode != 0 or not out_path.exists():
            print("RIFE: failed (returncode=%s, stderr=%s)" % (proc.returncode, (proc.stderr or "")[:500]), flush=True)
            return False
        return True

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cache_bgr: Dict[float, Path] = {0.0: tmp_path / "bgr0.png", 1.0: tmp_path / "bgr1.png"}
        cache_alpha: Dict[float, Path] = {0.0: tmp_path / "a0.png", 1.0: tmp_path / "a1.png"}
        cv2.imwrite(str(cache_bgr[0.0]), comp1)
        cv2.imwrite(str(cache_bgr[1.0]), comp2)
        cv2.imwrite(str(cache_alpha[0.0]), alpha1)
        cv2.imwrite(str(cache_alpha[1.0]), alpha2)

        def get_frame(t: float) -> Optional[tuple]:
            if t <= 0 or t >= 1:
                idx = 0.0 if t == 0 else 1.0
                return (cache_bgr[idx], cache_alpha[idx])
            if t in cache_bgr:
                return (cache_bgr[t], cache_alpha[t])
            if t == 0.5:
                p0_bgr, p1_bgr = cache_bgr[0.0], cache_bgr[1.0]
                p0_a, p1_a = cache_alpha[0.0], cache_alpha[1.0]
            elif t < 0.5:
                t1 = 2 * t
                if get_frame(t1) is None:
                    return None
                p0_bgr, p1_bgr = cache_bgr[0.0], cache_bgr[t1]
                p0_a, p1_a = cache_alpha[0.0], cache_alpha[t1]
            else:
                t0 = 2 * t - 1
                if get_frame(t0) is None:
                    return None
                p0_bgr, p1_bgr = cache_bgr[t0], cache_bgr[1.0]
                p0_a, p1_a = cache_alpha[t0], cache_alpha[1.0]
            key = int(round(t * 10000))
            out_bgr = tmp_path / f"bgr_{key}.png"
            out_a = tmp_path / f"a_{key}.png"
            if not rife_one(p0_bgr, p1_bgr, out_bgr) or not rife_one(p0_a, p1_a, out_a):
                return None
            cache_bgr[t] = out_bgr
            cache_alpha[t] = out_a
            return (out_bgr, out_a)

        results: List[bytes] = []
        for i in range(1, num_frames + 1):
            t = i / (num_frames + 1)
            t_dyadic = max(1 / 16.0, min(15 / 16.0, round(t * 16) / 16.0))
            pair = get_frame(t_dyadic)
            if pair is None:
                return None
            p_bgr, p_alpha = pair
            bgr_rifed = cv2.imread(str(p_bgr))
            alpha_rifed = cv2.imread(str(p_alpha), cv2.IMREAD_GRAYSCALE)
            if bgr_rifed.shape[:2] != alpha_rifed.shape[:2]:
                alpha_rifed = cv2.resize(alpha_rifed, (bgr_rifed.shape[1], bgr_rifed.shape[0]), interpolation=cv2.INTER_LINEAR)
            rgb_full = cv2.cvtColor(bgr_rifed, cv2.COLOR_BGR2RGB)
            _, alpha_clean = cv2.threshold(alpha_rifed, 50, 255, cv2.THRESH_TOZERO)
            mask_zero = alpha_clean == 0
            rgb_full[mask_zero] = [0, 0, 0]
            out_rgba = np.dstack((rgb_full, alpha_clean)).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(out_rgba, "RGBA").save(buf, format="PNG")
            results.append(buf.getvalue())
    return results


def _optical_flow_interpolate(
    frame1: np.ndarray, frame2: np.ndarray, alpha: float
) -> np.ndarray:
    """
    One intermediate frame at time alpha (0 < alpha < 1) using OpenCV optical flow.
    Fallback when RIFE is not available.
    """
    h, w = frame1.shape[:2]
    gray1 = cv2.cvtColor(frame1[:, :, :3], cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2[:, :, :3], cv2.COLOR_RGB2GRAY)
    flow_1_to_2 = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
    )
    flow_u = flow_1_to_2[:, :, 0]
    flow_v = flow_1_to_2[:, :, 1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xx + flow_u * alpha
    map_y = yy + flow_v * alpha
    out = np.zeros_like(frame1)
    for c in range(frame1.shape[2]):
        out[:, :, c] = cv2.remap(
            frame1[:, :, c], map_x, map_y,
            cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE,
        )
    return out


@app.post("/api/interpolate-frames")
async def interpolate_frames(
    first: UploadFile = File(..., description="First frame PNG"),
    last: UploadFile = File(..., description="Last frame PNG"),
    num_frames: int = Query(1, ge=1, le=20, description="Number of intermediate frames"),
    chroma_tolerance: int = Query(30, ge=0, le=100, description="Chroma tolerance (Step 3)"),
    chroma_holo_enabled: bool = Query(False, description="Holo remover enabled"),
    chroma_holo_strength: int = Query(80, ge=0, le=100, description="Holo remover strength"),
    chroma_max_clusters: int = Query(0, ge=0, le=100, description="Max clusters (0=all)"),
):
    """
    Generate intermediate frames between first and last using RIFE (rife-ncnn-vulkan)
    if available, otherwise OpenCV optical flow.
    Returns a ZIP containing: first.png, interp_001.png, ..., interp_N.png, last.png.
    """
    try:
        raw1 = await first.read()
        raw2 = await last.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read images: {e}")

    buf1 = io.BytesIO(raw1)
    buf2 = io.BytesIO(raw2)
    try:
        pil1 = Image.open(buf1).convert("RGBA")
        pil2 = Image.open(buf2).convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    arr1 = np.array(pil1)
    arr2 = np.array(pil2)
    if arr1.shape != arr2.shape:
        raise HTTPException(
            status_code=400,
            detail=f"Image size mismatch: {arr1.shape[:2]} vs {arr2.shape[:2]}",
        )

    interp_pngs: List[bytes] = []
    rife_results = _interpolate_with_rife_cli(
        pil1, pil2, num_frames,
        chroma_tolerance=chroma_tolerance,
        chroma_holo_enabled=chroma_holo_enabled,
        chroma_holo_strength=chroma_holo_strength,
        chroma_max_clusters=chroma_max_clusters,
    )
    if rife_results is not None and len(rife_results) == num_frames:
        print("Interpolation: using RIFE (rife-ncnn-vulkan), num_frames=%d" % num_frames, flush=True)
        interp_pngs = list(rife_results)
    if not interp_pngs:
        print("Interpolation: using OpenCV optical flow (RIFE not available), num_frames=%d" % num_frames, flush=True)
        for i in range(1, num_frames + 1):
            t = i / (num_frames + 1)
            interp = _optical_flow_interpolate(arr1, arr2, t)
            pil_interp = Image.fromarray(interp)
            buf_interp = io.BytesIO()
            pil_interp.save(buf_interp, format="PNG")
            interp_pngs.append(buf_interp.getvalue())

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        buf_first = io.BytesIO()
        pil1.save(buf_first, format="PNG")
        zf.writestr("first.png", buf_first.getvalue())
        for i, png_data in enumerate(interp_pngs):
            zf.writestr(f"interp_{i + 1:03d}.png", png_data)
        buf_last = io.BytesIO()
        pil2.save(buf_last, format="PNG")
        zf.writestr("last.png", buf_last.getvalue())

    zip_buf.seek(0)
    return Response(
        content=zip_buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=interpolated_frames.zip"},
    )


# Mount static assets (CSS, JS) if present
static_dir = PROJECT_ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
