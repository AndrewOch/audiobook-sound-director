"""FastAPI application for Audiobook Sound Director (packaged)."""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pathlib import Path
import shutil
from typing import Optional
import uuid
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Audiobook Sound Director",
    description="AI-powered sound direction for audiobooks",
    version="1.0.0"
)

# Setup directories
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"
OUTPUT_DIR = ROOT_DIR / "output"

# Create directories if they don't exist
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static and output directories
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# Setup templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/process")
async def process_audiobook(
    request: Request,
    input_type: str = Form(...),
    text_input: Optional[str] = Form(None),
    text_file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
):
    """Process audiobook input and generate sound direction.
    
    Args:
        input_type: Type of input ('text', 'text_file', 'audio_file')
        text_input: Direct text input
        text_file: Uploaded text file
        audio_file: Uploaded audio file
        
    Returns:
        JSON response with processing status and results
    """
    try:
        # Generate unique job ID and create job directory in output
        job_id = str(uuid.uuid4())
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract text based on input type
        text_content: Optional[str] = None
        
        if input_type == "text":
            if not text_input:
                raise HTTPException(status_code=400, detail="Text input is required")
            text_content = text_input
            
        elif input_type == "text_file":
            if not text_file:
                raise HTTPException(status_code=400, detail="Text file is required")
            # Read text file
            content = await text_file.read()
            text_content = content.decode("utf-8")
            
        elif input_type == "audio_file":
            if not audio_file:
                raise HTTPException(status_code=400, detail="Audio file is required")
            # Save audio file inside the job directory (no global uploads dir)
            audio_path = job_dir / f"input_{audio_file.filename}"
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio_file.file, f)
            
            # TODO: Transcribe audio using speech_recognition module
            text_content = "[Audio transcription will be implemented]"
            
        else:
            raise HTTPException(status_code=400, detail="Invalid input type")
        
        # TODO: Full processing pipeline steps (emotions, foli, music, mixer)
        
        # Mock response for now
        result = {
            "job_id": job_id,
            "status": "processing",
            "input_type": input_type,
            "text_preview": text_content[:200] if text_content else None,
            "timestamp": datetime.now().isoformat(),
            "message": "Processing started. Full pipeline will be implemented.",
            "steps": {
                "transcription": "pending" if input_type == "audio_file" else "skipped",
                "emotion_analysis": "pending",
                "foli_classification": "pending",
                "speech_generation": "pending" if input_type == "text" else "skipped",
                "music_generation": "pending",
                "foli_generation": "pending",
                "mixing": "pending"
            }
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    job_dir = OUTPUT_DIR / job_id
    
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    
    # TODO: Implement actual status tracking
    return JSONResponse(content={
        "job_id": job_id,
        "status": "completed",
        "files": {
            "mixed_audio": f"/output/{job_id}/mixed.wav",
            "speech_track": f"/output/{job_id}/speech.wav",
            "music_track": f"/output/{job_id}/music.wav",
            "foli_track": f"/output/{job_id}/foli.wav"
        }
    })


@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a generated audio file."""
    file_path = OUTPUT_DIR / job_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="audio/wav"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Audiobook Sound Director"}
