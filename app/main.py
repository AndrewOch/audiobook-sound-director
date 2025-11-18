"""FastAPI application for Audiobook Sound Director (packaged)."""

# КРИТИЧНО: Настраиваем кэш ПЕРЕД импортом любых модулей, которые могут использовать кэш
from pathlib import Path
import sys

# Определяем корень проекта
_APP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _APP_DIR.parent

# Настраиваем кэш на внешний диск ПЕРЕД загрузкой моделей
# Импортируем напрямую, чтобы избежать циклических зависимостей
from modules.cache_config import setup_cache_directories
setup_cache_directories(_ROOT_DIR)

# Теперь можно импортировать остальные модули
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi import BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import shutil
from typing import Optional
import uuid
from datetime import datetime

from modules.pipeline import PipelineService, PipelineServiceConfig
from modules.pipeline.dto import InputRequest, JobInfo, MixRequest, MixResponse
from modules.pipeline.registry import warm_up

# Initialize FastAPI app
app = FastAPI(
    title="Audiobook Sound Director",
    description="AI-powered sound direction for audiobooks",
    version="1.0.0"
)

# Setup directories
APP_DIR = _APP_DIR
ROOT_DIR = _ROOT_DIR
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


@app.on_event("startup")
async def _startup_preload():
    # Preload heavy models to avoid cold-start latency
    try:
        warm_up()
    except Exception:
        # Fail-safe: proceed even if some models fail to preload
        pass

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/process")
async def process_audiobook(
    request: Request,
    background_tasks: BackgroundTasks,
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
        # Create job meta and directory
        job_id = str(uuid.uuid4())
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job_info = JobInfo(job_id=job_id, job_dir=job_dir, created_at=datetime.now().isoformat())

        if input_type == "audio_file":
            if not audio_file:
                raise HTTPException(status_code=400, detail="Audio file is required")
            audio_path = job_dir / f"input_{audio_file.filename}"
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio_file.file, f)
        else:
            raise HTTPException(status_code=400, detail="На данном этапе поддерживается только ввод аудиофайла")

        # Write initial queued status file
        initial_steps = {
            "ingest": {"name": "ingest", "status": "pending"},
            "transcription": {"name": "transcription", "status": "pending"},
            "emotion_analysis": {"name": "emotion_analysis", "status": "pending"},
            "foli_classification": {"name": "foli_classification", "status": "pending"},
            "speech_generation": {"name": "speech_generation", "status": "skipped"},
            "music_generation": {"name": "music_generation", "status": "pending"},
            "foli_generation": {"name": "foli_generation", "status": "pending"},
            "mixing": {"name": "mixing", "status": "pending"},
        }
        status_payload = {
            "job_id": job_id,
            "status": "queued",
            "message": "Job queued",
            "timestamp": datetime.now().isoformat(),
            "steps": initial_steps,
            "outputs": {},
        }
        with open(job_dir / "job_status.json", "w", encoding="utf-8") as f:
            import json as _json
            _json.dump(status_payload, f, ensure_ascii=False, indent=2)

        # Schedule background execution
        def _run_pipeline_job(jid: str, apath: str):
            service_cfg = PipelineServiceConfig(
                enable_emotions=True,
                enable_foli_classification=True,
                enable_music_generation=True,
                enable_foli_generation=True,
                enable_mixing=False,
            )
            service = PipelineService(output_root=OUTPUT_DIR, config=service_cfg)
            info = JobInfo(job_id=jid, job_dir=OUTPUT_DIR / jid, created_at=datetime.now().isoformat())
            req = InputRequest(input_type="audio_file", audio_file_path=Path(apath))
            try:
                service.start_job(req, job=info, execute=True)
            except Exception as e:
                # Ensure we persist an error status even if the pipeline failed early
                try:
                    import json as _json
                    err_payload = {
                        "job_id": jid,
                        "status": "error",
                        "message": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "steps": {
                            "ingest": {"name": "ingest", "status": "error", "detail": str(e)},
                        },
                        "outputs": {},
                    }
                    with open((OUTPUT_DIR / jid / "job_status.json"), "w", encoding="utf-8") as _f:
                        _json.dump(err_payload, _f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

        background_tasks.add_task(_run_pipeline_job, job_id, str(audio_path))

        return JSONResponse(content={
            "job_id": job_id,
            "status": "queued",
            "message": "Processing started",
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tracks/{job_id}")
async def get_tracks(job_id: str):
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    tracks_path = job_dir / "tracks.json"
    if not tracks_path.exists():
        raise HTTPException(status_code=404, detail="Tracks not found")
    return FileResponse(path=tracks_path, media_type="application/json", filename="tracks.json")


@app.post("/api/mix")
async def mix_audio(request: Request):
    try:
        payload = await request.json()
        job_id = payload.get("job_id")
        tracks_settings = payload.get("tracks", [])
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id is required")

        job_dir = OUTPUT_DIR / job_id
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail="Job not found")

        tracks_json = job_dir / "tracks.json"
        if not tracks_json.exists():
            raise HTTPException(status_code=404, detail="Tracks not found for job")

        # Load track descriptors
        import json as _json
        with open(tracks_json, "r", encoding="utf-8") as f:
            track_descs = {t["id"]: t for t in _json.load(f)}

        # Build mixer TrackSpec list for enabled tracks
        from modules.mixer.config import TrackSpec as MixerTrackSpec
        from modules.mixer.mixer import AudioMixer

        def volume_to_db(v: float) -> float:
            # Map linear volume [0,1] to dB gain; clamp at -60 dB
            try:
                import math
                v = max(0.0, min(1.0, float(v)))
                if v <= 0.0005:
                    return -60.0
                return 20.0 * math.log10(v)
            except Exception:
                return 0.0

        specs = []
        for ts in tracks_settings:
            tid = ts.get("id")
            enabled = bool(ts.get("enabled", True))
            vol = float(ts.get("volume", 1.0))
            desc = track_descs.get(tid)
            if not desc or not enabled:
                continue
            specs.append(MixerTrackSpec(
                path=desc["path"],
                kind=desc["kind"],
                channel=desc.get("channel"),
                gain_db=volume_to_db(vol),
            ))

        if not specs:
            raise HTTPException(status_code=400, detail="No enabled tracks to mix")

        mixer = AudioMixer()
        output_path = job_dir / "mixed.wav"
        mixed_file = mixer.mix(specs, str(output_path))

        return JSONResponse(content=MixResponse(
            job_id=job_id,
            status="ok",
            download_url=f"/output/{job_id}/mixed.wav"
        ).__dict__)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(content=MixResponse(
            job_id=payload.get("job_id", ""),
            status="error",
            detail=str(e)
        ).__dict__, status_code=500)


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    job_dir = OUTPUT_DIR / job_id
    
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    status_path = job_dir / "job_status.json"
    if status_path.exists():
        return FileResponse(path=status_path, media_type="application/json", filename="job_status.json")
    # Fallback minimal info
    return JSONResponse(content={
        "job_id": job_id,
        "status": "processing",
        "timestamp": datetime.now().isoformat(),
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
