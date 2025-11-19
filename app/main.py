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
from fastapi import Body
from pydantic import BaseModel
import shutil
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime
import json as _json

from modules.pipeline import PipelineService, PipelineServiceConfig
from modules.pipeline.dto import InputRequest, JobInfo, MixRequest, MixResponse
from modules.pipeline.registry import warm_up
from modules.speech import get_elevenlabs_client, safe_filename

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


class ElevenLabsSpeechRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    model_id: Optional[str] = None
    job_id: Optional[str] = None
    filename: Optional[str] = None


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

        def _synthesize_text_to_speech(text: str) -> Dict[str, Any]:
            try:
                client = get_elevenlabs_client()
            except RuntimeError as err:
                raise HTTPException(status_code=500, detail=str(err))
            filename = safe_filename(prefix="speech", suffix=".mp3")
            output_path = job_dir / filename
            try:
                client.synthesize_to_file(text=text, output_path=output_path)
            except Exception as err:
                raise HTTPException(status_code=502, detail=f"ElevenLabs error: {err}")
            rel_url = f"/output/{output_path.relative_to(OUTPUT_DIR).as_posix()}"
            return {
                "path": str(output_path),
                "audio_url": rel_url,
                "filename": filename,
            }

        speech_info: Optional[Dict[str, Any]] = None

        if input_type == "text":
            text_value = (text_input or "").strip()
            if not text_value:
                raise HTTPException(status_code=400, detail="Text input is required")
            speech_info = _synthesize_text_to_speech(text_value)
            audio_path = Path(speech_info["path"])
        elif input_type == "text_file":
            if not text_file:
                raise HTTPException(status_code=400, detail="Text file is required")
            text_value = text_file.file.read().decode("utf-8", errors="ignore").strip()
            if not text_value:
                raise HTTPException(status_code=400, detail="Uploaded text file is empty")
            speech_info = _synthesize_text_to_speech(text_value)
            audio_path = Path(speech_info["path"])
        elif input_type == "audio_file":
            if not audio_file:
                raise HTTPException(status_code=400, detail="Audio file is required")
            audio_path = job_dir / f"input_{audio_file.filename}"
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio_file.file, f)
        else:
            raise HTTPException(status_code=400, detail="Unsupported input type")

        # Write initial queued status file
        initial_steps = {
            "ingest": {"name": "ingest", "status": "pending"},
            "transcription": {"name": "transcription", "status": "pending"},
            "emotion_analysis": {"name": "emotion_analysis", "status": "pending"},
            "foli_classification": {"name": "foli_classification", "status": "pending"},
            "speech_generation": {"name": "speech_generation", "status": "skipped"},
            "music_generation": {"name": "music_generation", "status": "skipped"},
            "foli_generation": {"name": "foli_generation", "status": "skipped"},
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
                enable_foli_classification=False,
                enable_music_generation=False,
                enable_foli_generation=False,
                enable_mixing=False,
            )
            service = PipelineService(output_root=OUTPUT_DIR, config=service_cfg)
            info = JobInfo(job_id=jid, job_dir=OUTPUT_DIR / jid, created_at=datetime.now().isoformat())
            req = InputRequest(input_type="audio_file", audio_file_path=Path(apath))
            try:
                service.start_job(req, job=info, execute=True)
            except Exception as e:
                error_file = job_dir / "error.log"
                try:
                    error_file.write_text(f"{datetime.now().isoformat()}\\n{str(e)}\\n", encoding="utf-8")
                except Exception:
                    pass
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
                        "outputs": {"error_log": f"/output/{jid}/{error_file.name}"} if error_file.exists() else {},
                    }
                    with open((OUTPUT_DIR / jid / "job_status.json"), "w", encoding="utf-8") as _f:
                        _json.dump(err_payload, _f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

        background_tasks.add_task(_run_pipeline_job, job_id, str(audio_path))

        response_payload = {
            "job_id": job_id,
            "status": "queued",
            "message": "Processing started",
            "timestamp": datetime.now().isoformat(),
        }
        if speech_info:
            response_payload["speech"] = {
                "filename": speech_info["filename"],
                "audio_url": speech_info["audio_url"],
            }

        return JSONResponse(content=response_payload)

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


@app.post("/api/speech/elevenlabs")
async def synthesize_speech_elevenlabs(payload: ElevenLabsSpeechRequest):
    """Generate speech with ElevenLabs and save the audio file in the output directory."""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        client = get_elevenlabs_client()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    target_dir = OUTPUT_DIR / (payload.job_id or "manual")
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = payload.filename or safe_filename()
    output_path = target_dir / filename

    try:
        audio_path = client.synthesize_to_file(
            text=text,
            output_path=output_path,
            voice_id=payload.voice_id,
            model_id=payload.model_id,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ElevenLabs error: {e}")

    relative_url = f"/output/{audio_path.relative_to(OUTPUT_DIR).as_posix()}"

    return JSONResponse(content={
        "status": "ok",
        "voice_id": payload.voice_id or client.config.voice_id,
        "model_id": payload.model_id or client.config.model_id,
        "filename": audio_path.name,
        "job_id": payload.job_id or "manual",
        "audio_url": relative_url,
    })


# -----------------------------
# Project Management Endpoints
# -----------------------------

@app.get("/api/project/{job_id}")
async def get_project(job_id: str):
    """Get complete project state including segments, emotions, foli, and tracks."""
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    
    import json as _json
    
    # Load transcript with segments (prefer merged)
    transcript_path = job_dir / "transcript_merged.json"
    if not transcript_path.exists():
        transcript_path = job_dir / "transcript.json"
    segments = []
    duration = 0.0
    if transcript_path.exists():
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_data = _json.load(f)
            segments = transcript_data.get("segments", [])
            duration = transcript_data.get("duration", 0.0)
    
    # Load segment emotions
    segments_emotions_path = job_dir / "segments_emotions.json"
    segments_emotions = {}
    if segments_emotions_path.exists():
        with open(segments_emotions_path, "r", encoding="utf-8") as f:
            emotions_list = _json.load(f)
            for emo in emotions_list:
                segments_emotions[emo["segment_id"]] = emo
    
    # Load segment foli
    segments_foli_path = job_dir / "segments_foli.json"
    segments_foli = {}
    if segments_foli_path.exists():
        with open(segments_foli_path, "r", encoding="utf-8") as f:
            foli_list = _json.load(f)
            for foli in foli_list:
                segments_foli[foli["segment_id"]] = foli
    
    # Merge segments with emotions and foli
    enriched_segments = []
    for idx, seg in enumerate(segments):
        # Use merged index as segment_id
        seg_id = idx
        enriched_seg = {
            "id": idx,  # Always use index for frontend
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", ""),
            "emotion": segments_emotions.get(seg_id, {}).get("emotion", "neutral"),
            "emotion_confidence": segments_emotions.get(seg_id, {}).get("confidence", 0.0),
            "foli_class": segments_foli.get(seg_id, {}).get("foli_class"),
            "foli_confidence": segments_foli.get(seg_id, {}).get("foli_confidence"),
            "foli": segments_foli.get(seg_id, {}).get("channels"),
        }
        enriched_segments.append(enriched_seg)
    
    # Load tracks
    tracks = []
    tracks_path = job_dir / "tracks.json"
    if tracks_path.exists():
        with open(tracks_path, "r", encoding="utf-8") as f:
            tracks = _json.load(f)
    
    # Get audio URL
    speech_wav = job_dir / "speech.wav"
    audio_url = f"/output/{job_id}/speech.wav" if speech_wav.exists() else None
    
    project_state = {
        "job_id": job_id,
        "audio_url": audio_url,
        "duration": duration,
        "segments": enriched_segments,
        "tracks": tracks,
        "playhead_position": 0.0,
        "zoom_level": 1.0,
    }
    
    return JSONResponse(content=project_state)


@app.get("/api/project/{job_id}/segments")
async def get_segments(job_id: str):
    """Get segments with emotions and foli classifications."""
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    
    import json as _json
    
    # Load transcript (prefer merged)
    transcript_path = job_dir / "transcript_merged.json"
    if not transcript_path.exists():
        transcript_path = job_dir / "transcript.json"
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = _json.load(f)
        segments = transcript_data.get("segments", [])
    
    # Load emotions
    segments_emotions = {}
    segments_emotions_path = job_dir / "segments_emotions.json"
    if segments_emotions_path.exists():
        with open(segments_emotions_path, "r", encoding="utf-8") as f:
            emotions_list = _json.load(f)
            for emo in emotions_list:
                segments_emotions[emo["segment_id"]] = emo
    
    # Load foli
    segments_foli = {}
    segments_foli_path = job_dir / "segments_foli.json"
    if segments_foli_path.exists():
        with open(segments_foli_path, "r", encoding="utf-8") as f:
            foli_list = _json.load(f)
            for foli in foli_list:
                segments_foli[foli["segment_id"]] = foli
    
    # Merge data
    enriched_segments = []
    for idx, seg in enumerate(segments):
        enriched_seg = {
            "id": idx,
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", ""),
            "emotion": segments_emotions.get(idx, {}).get("emotion", "neutral"),
            "emotion_confidence": segments_emotions.get(idx, {}).get("confidence", 0.0),
            "foli_class": segments_foli.get(idx, {}).get("foli_class"),
            "foli_confidence": segments_foli.get(idx, {}).get("foli_confidence"),
            "foli": segments_foli.get(idx, {}).get("channels"),
        }
        enriched_segments.append(enriched_seg)
    
    return JSONResponse(content={"segments": enriched_segments})


@app.put("/api/project/{job_id}/segments/emotions")
async def update_segment_emotions(job_id: str, request: Request):
    """Update emotions for one or more segments.
    
    Request body should be:
    {
        "updates": [
            {"segment_id": 0, "emotion": "happy", "confidence": 0.95},
            {"segment_id": 1, "emotion": "sad", "confidence": 0.87}
        ]
    }
    """
    try:
        payload = await request.json()
        updates = payload.get("updates", [])
        
        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")
        
        job_dir = OUTPUT_DIR / job_id
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        import json as _json
        
        # Load existing emotions
        segments_emotions_path = job_dir / "segments_emotions.json"
        segments_emotions_list = []
        
        if segments_emotions_path.exists():
            with open(segments_emotions_path, "r", encoding="utf-8") as f:
                segments_emotions_list = _json.load(f)
        
        # Convert to dict for easier updates
        emotions_dict = {emo["segment_id"]: emo for emo in segments_emotions_list}
        
        # Load transcript to get segment info if needed
        transcript_path = job_dir / "transcript.json"
        segments = []
        if transcript_path.exists():
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = _json.load(f)
                segments = transcript_data.get("segments", [])
        
        # Apply updates
        updated_count = 0
        for update in updates:
            seg_id = update.get("segment_id")
            if seg_id is None:
                continue
            
            # Get or create emotion entry
            if seg_id in emotions_dict:
                emo_entry = emotions_dict[seg_id]
            else:
                # Create new entry if segment exists
                if seg_id < len(segments):
                    seg = segments[seg_id]
                    emo_entry = {
                        "segment_id": seg_id,
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "text": seg.get("text", ""),
                        "emotion": "neutral",
                        "confidence": 0.0,
                        "top5": [],
                    }
                else:
                    continue  # Skip invalid segment_id
            
            # Update emotion and confidence
            if "emotion" in update:
                emo_entry["emotion"] = update["emotion"]
            if "confidence" in update:
                emo_entry["confidence"] = float(update["confidence"])
            
            emotions_dict[seg_id] = emo_entry
            updated_count += 1
        
        # Convert back to list and save
        segments_emotions_list = [emotions_dict[seg_id] for seg_id in sorted(emotions_dict.keys())]
        
        with open(segments_emotions_path, "w", encoding="utf-8") as f:
            _json.dump(segments_emotions_list, f, ensure_ascii=False, indent=2)
        
        return JSONResponse(content={
            "status": "ok",
            "message": f"Updated {updated_count} segment(s)",
            "updated_count": updated_count,
        })
    
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "detail": str(e)},
            status_code=500
        )


@app.post("/api/project/{job_id}/generate-segments")
async def generate_segments(job_id: str, request: Request):
    """Generate music or foli sounds for selected segments."""
    try:
        payload = await request.json()
        segment_ids = payload.get("segment_ids", [])
        generate_type = payload.get("type", "music")  # "music" or "foli"
        
        if not segment_ids:
            raise HTTPException(status_code=400, detail="No segments selected")
        
        job_dir = OUTPUT_DIR / job_id
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        import json as _json
        
        # Load segments with emotions/foli (prefer merged transcript)
        transcript_path = job_dir / "transcript_merged.json"
        if not transcript_path.exists():
            transcript_path = job_dir / "transcript.json"
        segments_emotions_path = job_dir / "segments_emotions.json"
        segments_foli_path = job_dir / "segments_foli.json"
        
        segments = []
        segments_emotions = {}
        segments_foli = {}
        
        if transcript_path.exists():
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = _json.load(f)
                segments = transcript_data.get("segments", [])
        
        if segments_emotions_path.exists():
            with open(segments_emotions_path, "r", encoding="utf-8") as f:
                emotions_list = _json.load(f)
                for emo in emotions_list:
                    segments_emotions[emo["segment_id"]] = emo
        
        if segments_foli_path.exists():
            with open(segments_foli_path, "r", encoding="utf-8") as f:
                foli_list = _json.load(f)
                for foli in foli_list:
                    segments_foli[foli["segment_id"]] = foli
        
        # Generate for each selected segment
        generated_tracks = []
        from modules.pipeline.registry import get_music_generator
        
        for seg_id in segment_ids:
            if seg_id >= len(segments):
                continue
            
            seg = segments[seg_id]
            seg_emotion = segments_emotions.get(seg_id, {})
            seg_foli = segments_foli.get(seg_id, {})
            
            if generate_type == "music":
                # Generate music based on emotion
                generator = get_music_generator()
                emotions = [(seg_emotion.get("emotion", "neutral"), seg_emotion.get("confidence", 0.5))]
                duration = seg.get("end", 0.0) - seg.get("start", 0.0)
                duration = max(5.0, min(duration, 30.0))  # Clamp between 5-30 seconds
                
                audio = generator.generate_from_emotions(emotions, duration_seconds=int(duration))
                output_path = job_dir / f"music_segment_{seg_id}.wav"
                generator.save_audio(audio, output_path)
                
                generated_tracks.append({
                    "id": f"music_segment_{seg_id}",
                    "type": "music",
                    "url": f"/output/{job_id}/music_segment_{seg_id}.wav",
                    "start_time": seg.get("start", 0.0),
                    "volume": 0.5,
                    "enabled": True,
                })
            
            elif generate_type == "foli":
                # Generate foli based on classification
                from modules.foli_generation import FoliGenerator, FoliGenConfig
                foli_gen = FoliGenerator(FoliGenConfig())
                foli_gen.load_model()
                
                # Legacy path: use ch1 if channels present
                ch_label = (seg_foli.get('channels') or {}).get('ch1', {}).get('class') or seg_foli.get('foli_class') or 'ambient background'
                prompt = f"The sound of {ch_label}. High quality, clear."
                duration = seg.get("end", 0.0) - seg.get("start", 0.0)
                duration = max(5.0, min(duration, 30.0))
                
                audio = foli_gen.generate(
                    prompt=prompt,
                    audio_length_in_s=duration,
                )
                output_path = job_dir / f"foli_segment_{seg_id}_ch1.wav"
                foli_gen.save_audio(audio, output_path)
                
                generated_tracks.append({
                    "id": f"foli_segment_{seg_id}_ch1",
                    "type": "background",
                    "url": f"/output/{job_id}/foli_segment_{seg_id}_ch1.wav",
                    "start_time": seg.get("start", 0.0),
                    "volume": 0.6,
                    "enabled": True,
                })
        
        return JSONResponse(content={
            "status": "ok",
            "tracks": generated_tracks,
            "message": f"Generated {len(generated_tracks)} {generate_type} tracks"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "detail": str(e)},
            status_code=500
        )


@app.post("/api/project/{job_id}/save")
async def save_project(job_id: str, request: Request):
    """Save project state (for server-side storage if needed)."""
    try:
        payload = await request.json()
        job_dir = OUTPUT_DIR / job_id
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        import json as _json
        project_state_path = job_dir / "project_state.json"
        
        payload["updated_at"] = datetime.now().isoformat()
        
        with open(project_state_path, "w", encoding="utf-8") as f:
            _json.dump(payload, f, ensure_ascii=False, indent=2)
        
        return JSONResponse(content={"status": "ok", "message": "Project saved"})
    
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "detail": str(e)},
            status_code=500
        )


@app.get("/api/projects")
async def list_projects():
    """List all saved projects (from localStorage on client, this is optional server-side list)."""
    # This could scan OUTPUT_DIR for project_state.json files
    # For now, return empty list as projects are stored in localStorage
    projects = []
    
    # Optional: scan for projects with project_state.json
    if OUTPUT_DIR.exists():
        for job_dir in OUTPUT_DIR.iterdir():
            if job_dir.is_dir():
                project_state_path = job_dir / "project_state.json"
                if project_state_path.exists():
                    import json as _json
                    try:
                        with open(project_state_path, "r", encoding="utf-8") as f:
                            state = _json.load(f)
                            projects.append({
                                "job_id": state.get("job_id", job_dir.name),
                                "created_at": state.get("created_at"),
                                "updated_at": state.get("updated_at"),
                                "duration": state.get("duration", 0.0),
                            })
                    except Exception:
                        pass
    
    return JSONResponse(content={"projects": projects})


# -----------------------------
# Background generation tasks
# -----------------------------

def _tasks_file(job_dir: Path) -> Path:
    return job_dir / "generation_tasks.json"

def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return _json.load(f)
    except Exception:
        pass
    return default

def _write_json(path: Path, data: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _append_tracks(job_dir: Path, new_tracks: List[Dict[str, Any]]) -> None:
    tracks_path = job_dir / "tracks.json"
    tracks: List[Dict[str, Any]] = _read_json(tracks_path, default=[])
    # deduplicate by id: replace existing with same id
    existing_map: Dict[str, Dict[str, Any]] = {t["id"]: t for t in tracks if "id" in t}
    for t in new_tracks:
        existing_map[t["id"]] = t
    tracks = list(existing_map.values())
    _write_json(tracks_path, tracks)

def _run_generation_task(job_id: str, job_dir: Path, task_id: str, kind: str, segment_ids: List[int]) -> None:
    tasks_path = _tasks_file(job_dir)
    tasks_state: Dict[str, Any] = _read_json(tasks_path, default={})
    def update(state: Dict[str, Any]) -> None:
        tasks_state[task_id] = state
        _write_json(tasks_path, tasks_state)
    try:
        update({
            "task_id": task_id,
            "type": kind,
            "segment_ids": segment_ids,
            "state": "running",
            "progress": 0,
            "message": "Starting",
            "created_at": datetime.now().isoformat()
        })
        # Load segment data (prefer merged)
        transcript_path = job_dir / "transcript_merged.json"
        if not transcript_path.exists():
            transcript_path = job_dir / "transcript.json"
        segments_emotions_path = job_dir / "segments_emotions.json"
        segments_foli_path = job_dir / "segments_foli.json"
        transcript = _read_json(transcript_path, default={"segments": []})
        segments = transcript.get("segments", [])
        seg_emotions_list = _read_json(segments_emotions_path, default=[])
        seg_foli_list = _read_json(segments_foli_path, default=[])
        seg_emotions = {e.get("segment_id"): e for e in seg_emotions_list}
        seg_foli = {f.get("segment_id"): f for f in seg_foli_list}

        generated_tracks: List[Dict[str, Any]] = []
        total = max(1, len(segment_ids))

        if kind == "music":
            from modules.pipeline.registry import get_music_generator
            music_gen = get_music_generator()
        elif kind == "foli":
            from modules.foli_generation import FoliGenerator, FoliGenConfig
            foli_gen = FoliGenerator(FoliGenConfig())
            foli_gen.load_model()
        else:
            raise ValueError("Unsupported generation type")

        # Special handling: for music with multiple segments -> generate ONE continuous track
        if kind == "music" and len(segment_ids) > 1:
            try:
                sids = sorted([sid for sid in segment_ids if sid < len(segments)])
                if not sids:
                    raise ValueError("No valid segments selected")
                first_id = sids[0]
                last_id = sids[-1]
                first_seg = segments[first_id]
                last_seg = segments[last_id]
                start_time = float(first_seg.get("start", 0.0))
                end_time = float(last_seg.get("end", 0.0))
                duration = max(5.0, float(end_time - start_time))
                # Cap duration to avoid extreme requests
                duration = min(duration, 120.0)
                # Build timeline prompt with emotions across selected range
                lines = []
                for sid in sids:
                    seg = segments[sid]
                    emo = seg_emotions.get(sid, {})
                    label = emo.get("emotion", "neutral")
                    conf = float(emo.get("confidence", 0.0))
                    rel_start = max(0.0, float(seg.get("start", 0.0)) - start_time)
                    rel_end = max(rel_start, float(seg.get("end", 0.0)) - start_time)
                    lines.append(f"{rel_start:.1f}-{rel_end:.1f}s: {label} ({int(conf*100)}%)")
                timeline_text = "\n".join(lines)
                prompt = (
                    f"Generate a cohesive background music for audiobook narration. "
                    f"Total duration: {int(duration)} seconds. Use smooth transitions when emotion changes. "
                    f"Timeline (relative to start):\n{timeline_text}"
                )
                audio = music_gen.generate_from_prompt(prompt, duration_seconds=int(duration))
                out_path = job_dir / f"music_range_{first_id}_{last_id}.wav"
                music_gen.save_audio(audio, out_path)
                generated_tracks.append({
                    "id": f"music_range_{first_id}_{last_id}",
                    "type": "music",
                    "url": f"/output/{job_id}/music_range_{first_id}_{last_id}.wav",
                    "start_time": start_time,
                    "volume": 0.5,
                    "enabled": True,
                })
                _append_tracks(job_dir, generated_tracks)
                update({
                    "task_id": task_id,
                    "type": kind,
                    "segment_ids": segment_ids,
                    "state": "completed",
                    "progress": 100,
                    "message": f"Completed 1 track for range {first_id}-{last_id}",
                    "result_tracks": generated_tracks,
                    "created_at": tasks_state.get(task_id, {}).get("created_at"),
                    "finished_at": datetime.now().isoformat(),
                })
                return
            except Exception as gen_err:
                update({
                    "task_id": task_id,
                    "type": kind,
                    "segment_ids": segment_ids,
                    "state": "error",
                    "progress": 0,
                    "message": f"Music range generation failed: {gen_err}",
                    "created_at": tasks_state.get(task_id, {}).get("created_at"),
                    "finished_at": datetime.now().isoformat(),
                })
                return

        for idx, seg_id in enumerate(segment_ids):
            if seg_id >= len(segments):
                continue
            seg = segments[seg_id]
            start_time = float(seg.get("start", 0.0))
            duration = float(seg.get("end", 0.0)) - start_time
            duration = max(5.0, min(duration, 30.0))

            if kind == "music":
                emo = seg_emotions.get(seg_id, {})
                emotions = [(emo.get("emotion", "neutral"), float(emo.get("confidence", 0.5)))]
                audio = music_gen.generate_from_emotions(emotions, duration_seconds=int(duration))
                out_path = job_dir / f"music_segment_{seg_id}.wav"
                music_gen.save_audio(audio, out_path)
                generated_tracks.append({
                    "id": f"music_segment_{seg_id}",
                    "type": "music",
                    "url": f"/output/{job_id}/music_segment_{seg_id}.wav",
                    "start_time": start_time,
                    "volume": 0.5,
                    "enabled": True,
                })
            else:
                ff = seg_foli.get(seg_id, {})
                # Determine channel for prompt
                tasks_state_here = _read_json(tasks_path, default={})
                channel = tasks_state_here.get(task_id, {}).get("foli_channel") or "ch1"
                channels = ff.get("channels") or {}
                label = (channels.get(channel) or {}).get("class") or ff.get("foli_class") or "ambient background"
                # Skip generation if channel is Silence
                if isinstance(label, str) and label.lower() == "silence":
                    update({
                        "task_id": task_id,
                        "type": kind,
                        "segment_ids": segment_ids,
                        "state": "running",
                        "progress": int(((idx + 1) / total) * 100),
                        "message": f"Skipped seg {seg_id} ({channel}=Silence)",
                        "created_at": tasks_state.get(task_id, {}).get("created_at"),
                    })
                    continue
                prompt = f"The sound of {label}. High quality, clear."
                audio = foli_gen.generate(prompt=prompt, audio_length_in_s=duration)
                out_path = job_dir / f"foli_segment_{seg_id}_{channel}.wav"
                foli_gen.save_audio(audio, out_path)
                generated_tracks.append({
                    "id": f"foli_segment_{seg_id}_{channel}",
                    "type": "background",
                    "url": f"/output/{job_id}/foli_segment_{seg_id}_{channel}.wav",
                    "start_time": start_time,
                    "volume": 0.6,
                    "enabled": True,
                })
            update({
                "task_id": task_id,
                "type": kind,
                "segment_ids": segment_ids,
                "state": "running",
                "progress": int(((idx + 1) / total) * 100),
                "message": f"Generated {idx+1}/{total}",
                "created_at": tasks_state.get(task_id, {}).get("created_at"),
            })

        # Persist tracks
        _append_tracks(job_dir, generated_tracks)
        update({
            "task_id": task_id,
            "type": kind,
            "segment_ids": segment_ids,
            "state": "completed",
            "progress": 100,
            "message": f"Completed {len(generated_tracks)} tracks",
            "result_tracks": generated_tracks,
            "created_at": tasks_state.get(task_id, {}).get("created_at"),
            "finished_at": datetime.now().isoformat(),
        })
    except Exception as e:
        update({
            "task_id": task_id,
            "type": kind,
            "segment_ids": segment_ids,
            "state": "error",
            "progress": 0,
            "message": str(e),
            "created_at": tasks_state.get(task_id, {}).get("created_at"),
            "finished_at": datetime.now().isoformat(),
        })

@app.post("/api/project/{job_id}/tasks/generate")
async def create_generation_task(job_id: str, request: Request, background_tasks: BackgroundTasks):
    try:
        payload = await request.json()
        segment_ids = payload.get("segment_ids", [])
        kind = payload.get("type", "music")
        foli_channel = payload.get("foli_channel")
        if not segment_ids:
            raise HTTPException(status_code=400, detail="segment_ids is required")
        job_dir = OUTPUT_DIR / job_id
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        task_id = f"t-{uuid.uuid4()}"
        # create task entry
        tasks_path = _tasks_file(job_dir)
        tasks_state: Dict[str, Any] = _read_json(tasks_path, default={})
        tasks_state[task_id] = {
            "task_id": task_id,
            "type": kind,
            "segment_ids": segment_ids,
            "state": "queued",
            "progress": 0,
            "message": "Queued",
            "created_at": datetime.now().isoformat(),
            "foli_channel": foli_channel,
        }
        _write_json(tasks_path, tasks_state)
        # enqueue background processing
        background_tasks.add_task(_run_generation_task, job_id, job_dir, task_id, kind, segment_ids)
        return JSONResponse(content={"status": "ok", "task_id": task_id})
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(content={"status": "error", "detail": str(e)}, status_code=500)

@app.get("/api/project/{job_id}/tasks/{task_id}")
async def get_generation_task(job_id: str, task_id: str):
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    tasks_path = _tasks_file(job_dir)
    tasks_state: Dict[str, Any] = _read_json(tasks_path, default={})
    task = tasks_state.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(content=task)

@app.get("/api/project/{job_id}/tasks")
async def list_generation_tasks(job_id: str):
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    tasks_path = _tasks_file(job_dir)
    tasks_state: Dict[str, Any] = _read_json(tasks_path, default={})
    return JSONResponse(content={"tasks": list(tasks_state.values())})
