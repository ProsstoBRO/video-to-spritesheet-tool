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

## Step 1: Upload & Preview

- Drag and drop or click to upload a video file (MP4, AVI, MOV, WebM)
- Preview the uploaded video in the player
- Use "Change video" to select a different file
