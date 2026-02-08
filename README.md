# Video to Sprite Sheet

Convert video files to sprite sheets with chroma keying, frame sampling, and optional looping.

## Setup

```bash
# Create virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0
```

Open http://localhost:8000 in your browser.

### Optional: RIFE for better loop interpolation (Step 5)

For higher-quality intermediate frames between the first and last frame (smoother looping), you can use **rife-ncnn-vulkan**:

1. Download the release for your OS: [rife-ncnn-vulkan releases](https://github.com/nihui/rife-ncnn-vulkan/releases)
2. Extract the archive **into the project root** so you get:
   ```
   video-to-spritesheet/
     rife-ncnn-vulkan/   (or rife-ncnn-vulkan-win64/)
       rife-ncnn-vulkan.exe
       models/
     app/
     static/
     ...
   ```
3. No PATH or environment variables needed — the app will find it automatically.

Alternatively, you can set `RIFE_NCNN_VULKAN` to the folder or exe path. If RIFE is not found, the app falls back to OpenCV optical flow.

## Step 1: Upload & Preview

- Drag and drop or click to upload a video file (MP4, AVI, MOV, WebM)
- Preview the uploaded video in the player
- Use "Change video" to select a different file
