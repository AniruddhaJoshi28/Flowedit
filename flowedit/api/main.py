"""
FlowEdit FastAPI REST Service.

Provides endpoints for:
- /api/transcribe: Speech-to-Text transcription via Whisper
- /api/correct: Learn a pronunciation correction from reference audio
- /api/synthesize: Synthesize speech with automatic Hopfield memory retrieval
- /api/baseline: Vanilla baseline synthesis without memory
- /api/memory: Inspect or clear stored associative memory
"""

import os
import json
import asyncio
import shutil
import tempfile
import subprocess
import traceback
import logging

# Ensure environment variables from .env are loaded before any FlowEdit module imports
from flowedit.utils.env import load_flowedit_env
load_flowedit_env()
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

import torch
import numpy as np
try:
    import soundfile as sf
except ImportError:
    sf = None


def write_audio_file(path: str, data: np.ndarray, sample_rate: int):
    """Write audio to WAV file using soundfile or built-in wave module."""
    if sf is not None:
        sf.write(path, data, sample_rate)
    else:
        import wave
        int16_data = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1 if data.ndim == 1 else data.shape[1])
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_data.tobytes())

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from flowedit.config import FlowEditConfig
from flowedit.audio.prompt_validator import ReferenceAudioError
from flowedit.pipeline.correction_loop import CorrectionLoop
from flowedit.pipeline.inference import FlowEditInference
from flowedit.alignment.whisper_aligner import WhisperAligner
from flowedit.memory.s3_storage import s3_spelling_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("flowedit.api")

# Global instances
correction_pipeline: Optional[CorrectionLoop] = None
inference_pipeline: Optional[FlowEditInference] = None
MEMORY_PATH = "./corrections.pt"


def convert_to_wav(input_path: str) -> str:
    """Ensure any uploaded audio file is a strict 24kHz Mono WAV."""
    output_path = input_path + "_converted.wav"
    # 1. Try ffmpeg if available on system
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            if os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except Exception:
                    pass
            return output_path
    except Exception:
        pass

    # 2. Pure Python / Torch fallback (no ffmpeg dependency required)
    try:
        import wave
        with wave.open(input_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            if n_frames > 0:
                raw_bytes = wf.readframes(n_frames)
                dtype = np.int16 if sampwidth == 2 else np.int32
                audio_np = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)
                if n_channels > 1:
                    audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)
                audio_np /= (32768.0 if dtype == np.int16 else 2147483648.0)

                # Resample to 24,000 Hz if needed
                if framerate != 24000:
                    t_in = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
                    target_len = max(1, int(round(len(audio_np) * (24000.0 / float(framerate)))))
                    t_out = torch.nn.functional.interpolate(t_in, size=target_len, mode="linear", align_corners=False)
                    audio_np = t_out.squeeze().numpy()

                out_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)
                with wave.open(output_path, "wb") as out_wf:
                    out_wf.setnchannels(1)
                    out_wf.setsampwidth(2)
                    out_wf.setframerate(24000)
                    out_wf.writeframes(out_int16.tobytes())
                if os.path.exists(input_path):
                    try:
                        os.remove(input_path)
                    except Exception:
                        pass
                return output_path
    except Exception as e_py:
        logger.debug(f"Pure Python audio conversion failed: {e_py}")

    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass
    return input_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    global correction_pipeline, inference_pipeline
    logger.info("Starting up FlowEdit API...")
    config = FlowEditConfig()
    
    # Configure paths from environment if provided
    config.backbone.backbone_type = os.environ.get("FLOWEDIT_BACKBONE_TYPE", "xtts")
    config.backbone.xtts_model_dir = os.environ.get("FLOWEDIT_XTTS_DIR", os.environ.get("FLOWEDIT_MODEL_DIR", ""))
    config.backbone.xtts_checkpoint = os.environ.get("FLOWEDIT_XTTS_CKPT", "model.pth")
    config.backbone.f5tts_ckpt_file = os.environ.get("FLOWEDIT_F5TTS_CKPT", "")
    config.backbone.f5tts_vocab_file = os.environ.get("FLOWEDIT_F5TTS_VOCAB", "")
    config.backbone.vocoder_local_path = os.environ.get("FLOWEDIT_VOCODER_DIR", "")
    whisper_model_env = os.environ.get("FLOWEDIT_WHISPER_MODEL", "")
    if whisper_model_env:
        config.alignment.whisper_model = whisper_model_env

    correction_pipeline = CorrectionLoop(config)
    correction_pipeline.load_models()

    if os.path.exists(MEMORY_PATH):
        try:
            correction_pipeline.memory.load(MEMORY_PATH)
        except Exception as e:
            logger.warning(f"Could not load memory from {MEMORY_PATH}: {e}")

    inference_pipeline = FlowEditInference(config)
    inference_pipeline.load(
        backbone=correction_pipeline.backbone,
        memory=correction_pipeline.memory,
    )

    logger.info(f"FlowEdit API ready. Loaded memory entries: {correction_pipeline.memory.num_entries}")
    yield
    logger.info("Shutting down FlowEdit API...")


app = FastAPI(
    title="FlowEdit API",
    description="<h3>👉 <a href='/' style='color:#DC2626; font-weight:bold;'>Click here to Open FlowEdit Web UI</a></h3><p>Lifelong Pronunciation Adaptation for Flow-Matching TTS via Associative Memory (arXiv:2606.20518)</p>",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "static"))


# ==============================================================================
# DIRECT HARDCODED VOICE PATHS (Set your exact .wav paths here)
# ==============================================================================
MICHAEL_VOICE_FILE_PATH: Optional[str] = None   # e.g. r"C:\Users\IOTPL\Project\flow_edit\michael.wav"
BLESSING_VOICE_FILE_PATH: Optional[str] = None  # e.g. r"C:\Users\IOTPL\Project\flow_edit\blessing.wav"

# Hardcoded default deployment voice mappings
HARDCODED_DEFAULT_VOICES = {
    "female": "blessing",
    "woman": "blessing",
    "blessing": "blessing",
    "blessing.wav": "blessing",
    "male": "michael",
    "man": "michael",
    "michael": "michael",
    "michael.wav": "michael",
}


def get_hardcoded_voice_path(target_name: str) -> Optional[str]:
    """Find the exact hardcoded path for blessing or michael across project candidate directories."""
    if not target_name:
        target_name = "blessing"

    # If target_name is already a valid file path on disk, return its absolute path directly
    if os.path.isfile(target_name) and os.path.getsize(target_name) > 1000:
        return os.path.abspath(target_name)

    target_lower = str(target_name).lower().strip()
    is_male = "male" in target_lower or "michael" in target_lower or "man" in target_lower

    # 1. Direct explicit file path if configured above
    if is_male and MICHAEL_VOICE_FILE_PATH and os.path.isfile(MICHAEL_VOICE_FILE_PATH):
        return os.path.abspath(MICHAEL_VOICE_FILE_PATH)
    if not is_male and BLESSING_VOICE_FILE_PATH and os.path.isfile(BLESSING_VOICE_FILE_PATH):
        return os.path.abspath(BLESSING_VOICE_FILE_PATH)

    voice_key = HARDCODED_DEFAULT_VOICES.get(target_lower, "michael" if is_male else "blessing")
    filename = f"{voice_key}.wav"

    pkg_api_dir = os.path.dirname(os.path.abspath(__file__))
    flowedit_root = os.path.abspath(os.path.join(pkg_api_dir, "..", ".."))
    workspace_root = os.path.abspath(os.path.join(flowedit_root, ".."))

    candidate_locations = [
        os.path.join(flowedit_root, "deploy_voices", filename),
        os.path.abspath(os.path.join(pkg_api_dir, "..", "resources", filename)),
        os.path.join(workspace_root, "deploy_voices", filename),
        os.path.join(workspace_root, filename),
        os.path.join(flowedit_root, filename),
        os.path.join(os.getcwd(), "deploy_voices", filename),
        os.path.join(os.getcwd(), filename),
    ]

    env_dir = os.environ.get("FLOWEDIT_VOICES_DIR", "")
    if env_dir:
        candidate_locations.insert(0, os.path.join(env_dir, filename))

    for cand in candidate_locations:
        if cand and os.path.isfile(cand) and os.path.getsize(cand) > 1000:
            return os.path.abspath(cand)

    # Fallback to default_speaker.wav in resources if blessing requested
    if voice_key == "blessing":
        res_default = os.path.abspath(os.path.join(pkg_api_dir, "..", "resources", "default_speaker.wav"))
        if os.path.isfile(res_default) and os.path.getsize(res_default) > 1000:
            return res_default

    return None


def get_preset_voices() -> Dict[str, str]:
    """Find preset deployment voice WAV files (Blessing for female, Michael for male)."""
    voices: Dict[str, str] = {}

    # Explicit hardcoded registration
    blessing_path = get_hardcoded_voice_path("blessing")
    if blessing_path:
        voices["blessing"] = blessing_path

    michael_path = get_hardcoded_voice_path("michael")
    if michael_path:
        voices["michael"] = michael_path

    # Search any additional voice files in deploy folders
    env_dir = os.environ.get("FLOWEDIT_VOICES_DIR", "")
    pkg_api_dir = os.path.dirname(os.path.abspath(__file__))
    flowedit_root = os.path.abspath(os.path.join(pkg_api_dir, "..", ".."))
    workspace_root = os.path.abspath(os.path.join(flowedit_root, ".."))

    dedicated_dirs = [
        os.path.join(flowedit_root, "deploy_voices"),
        os.path.join(workspace_root, "deploy_voices"),
        env_dir,
    ]

    for d in dedicated_dirs:
        if d and os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".wav"):
                    name = os.path.splitext(f)[0].lower()
                    if name not in voices:
                        cand_f = os.path.abspath(os.path.join(d, f))
                        if os.path.isfile(cand_f) and os.path.getsize(cand_f) > 1000:
                            voices[name] = cand_f

    return voices


async def resolve_speaker_path(speaker_wav: Optional[UploadFile], speaker_name: Optional[str]) -> tuple[str, bool]:
    """Resolves speaker WAV file path from upload or hardcoded preset name. Returns (path, is_temp)."""
    # 1. Only process speaker_wav if a non-empty audio file was genuinely uploaded
    if speaker_wav is not None and getattr(speaker_wav, "filename", None):
        try:
            content = await speaker_wav.read()
            # Must be non-empty audio with valid header (> 100 bytes)
            if content and len(content) > 100:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
                    temp_path = temp_speaker.name
                with open(temp_path, "wb") as f:
                    f.write(content)
                converted_path = convert_to_wav(temp_path)
                return converted_path, True
            else:
                logger.info("Empty speaker_wav upload received; defaulting to hardcoded preset voice.")
        except Exception as e:
            logger.warning(f"Error reading uploaded speaker_wav: {e}. Falling back to hardcoded voice.")

    # 2. Check if speaker_name is already a valid file path on disk
    if speaker_name and os.path.isfile(speaker_name) and os.path.getsize(speaker_name) > 100:
        return os.path.abspath(speaker_name), False

    # 3. Hardcoded voice resolution: female -> blessing, male -> michael
    name_key = (speaker_name or "female").strip().lower()
    if name_key in ("male", "man", "michael"):
        target_name = "michael"
    else:
        target_name = "blessing"  # Default female voice (Blessing)

    voice_path = get_hardcoded_voice_path(target_name)
    if voice_path and os.path.isfile(voice_path):
        return voice_path, False

    # Check preset voices dictionary
    voices = get_preset_voices()
    if target_name in voices and os.path.isfile(voices[target_name]):
        return voices[target_name], False

    # Alternate gender fallback
    alt_target = "michael" if target_name == "blessing" else "blessing"
    alt_path = get_hardcoded_voice_path(alt_target)
    if alt_path and os.path.isfile(alt_path):
        return alt_path, False

    # Failsafe fallback to any available wav file
    if voices:
        return next(iter(voices.values())), False

    bundled_default = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "default_speaker.wav"))
    if os.path.isfile(bundled_default):
        return bundled_default, False

    raise HTTPException(status_code=400, detail="No speaker reference voice provided or found.")


from flowedit.api.ui_html import get_ui_html


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/ui", response_class=HTMLResponse, summary="FlowEdit Web Interface")
def read_root():
    """Serve the FlowEdit Red & White web UI."""
    return HTMLResponse(content=get_ui_html())


@app.get("/api/status")
async def get_status():
    """Get system runtime and memory status."""
    global correction_pipeline
    if not correction_pipeline:
        return {"status": "initializing"}
    bb = correction_pipeline.backbone
    return {
        "status": "ready",
        "backbone_type": getattr(bb.config, "backbone_type", "xtts"),
        "device": str(bb.device),
        "embedding_dim": bb.embedding_dim,
        "memory_entries": correction_pipeline.memory.num_entries if correction_pipeline.memory else 0,
    }


@app.get("/api/speakers")
async def list_speakers():
    """List available preset speaker voices."""
    voices = get_preset_voices()
    result = []
    for name, path in voices.items():
        result.append({
            "name": name,
            "display_name": name.capitalize() + (" (Female Voice)" if name in ("blessing", "kokoro") else " (Male Voice)" if name == "michael" else ""),
            "path": path,
            "preview_url": f"/api/speakers/{name}/audio",
        })
    return {"speakers": result}


@app.get("/api/speakers/{name}/audio")
async def get_speaker_audio(name: str):
    """Preview a speaker voice audio."""
    voices = get_preset_voices()
    key = name.lower()
    if key in ("female", "woman"):
        key = "blessing" if "blessing" in voices else key
    elif key in ("male", "man"):
        key = "michael" if "michael" in voices else key
    if key in voices and os.path.isfile(voices[key]):
        return FileResponse(voices[key], media_type="audio/wav")
    raise HTTPException(status_code=404, detail=f"Speaker voice '{name}' not found.")


@app.exception_handler(ReferenceAudioError)
async def reference_audio_exception_handler(request: Request, exc: ReferenceAudioError):
    return JSONResponse(
        status_code=422,
        content={"code": getattr(exc, "code", "REFERENCE_AUDIO_INVALID"), "message": str(exc)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return JSONResponse(status_code=500, content={"detail": str(exc), "traceback": tb})


@app.post("/api/correct")
async def correct_pronunciation(
    text: str = Form(..., description="Full text containing target word"),
    target_word: str = Form(..., description="The word to correct pronunciation of"),
    language: str = Form("en", description="Language code"),
    mode: str = Form("audio", description="Correction mode: 'audio' (default, 3-stage FlowEdit optimization) or 'spell' (deterministic phonetic respelling)"),
    spell_as: Optional[str] = Form(None, description="Replacement phonetic spelling when mode='spell' (e.g. 'red' for 'read')"),
    ref_audio: Optional[UploadFile] = File(None, description="Reference audio with correct pronunciation (required for mode='audio')"),
    speaker_wav: Optional[UploadFile] = File(None, description="Speaker reference audio for voice conditioning"),
    speaker_name: Optional[str] = Form("female", description="Preset speaker voice name if not uploading audio"),
    ref_text: Optional[str] = Form(None, description="Optional transcription of speaker audio"),
    occurrence_index: int = Form(0, description="Occurrence index of target word if multiple exist"),
    phonetic_hint: Optional[str] = Form(None, description="Phonetic hint for warm-starting optimization"),
):
    """Pronunciation correction endpoint supporting two modes:
    1. 'audio': Full FlowEdit 3-stage continuous latent optimization from reference audio (untouched).
    2. 'spell': Deterministic phonetic respelling (e.g. 'read' -> 'red') stored directly in S3.
    """
    global correction_pipeline
    text = text.strip()
    target_word = target_word.strip()

    if not text or not target_word:
        raise HTTPException(status_code=400, detail="text and target_word must be non-empty.")

    mode_clean = (mode or "audio").strip().lower()

    # ─────────────────────────────────────────────────────────────────────────
    # Mode 2: 'spell' — Deterministic Phonetic Respelling with Direct S3 Sync
    # ─────────────────────────────────────────────────────────────────────────
    if mode_clean == "spell":
        replacement = (spell_as or "").strip()
        if not replacement:
            raise HTTPException(status_code=400, detail="spell_as is required when mode='spell'.")

        reg_result = s3_spelling_store.add_correction(
            word=target_word,
            spell_as=replacement,
            carrier_text=text,
            language=language,
        )

        return JSONResponse({
            "success": True,
            "mode": "spell",
            "word": target_word,
            "spell_as": replacement,
            "sense_id": reg_result.get("sense_id"),
            "sense_display": reg_result.get("sense_display"),
            "carrier_text": text,
            "s3_synced": reg_result.get("s3_synced", False),
            "s3_status": reg_result.get("s3_status", "pending_s3_link"),
            "s3_uri": reg_result.get("s3_uri"),
            "total_spelling_entries": reg_result.get("total_entries", 1),
            "memory_size": correction_pipeline.memory.num_entries if correction_pipeline and correction_pipeline.memory else 0,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Mode 1: 'audio' — Existing FlowEdit Latent Optimization (Untouched)
    # ─────────────────────────────────────────────────────────────────────────
    if not ref_audio:
        raise HTTPException(status_code=400, detail="ref_audio is required for audio correction mode.")

    if not correction_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref:
        temp_ref_path = temp_ref.name
    await ref_audio.seek(0)
    with open(temp_ref_path, "wb") as f:
        f.write(await ref_audio.read())
    temp_ref_path = convert_to_wav(temp_ref_path)

    temp_speaker_path, is_temp_speaker = await resolve_speaker_path(speaker_wav, speaker_name)

    # Collect available preset voices to include in multi-speaker joint optimization if requested
    preset_voices = get_preset_voices()
    extra_speakers = []
    if (speaker_name or "").lower() in ("all", "both", "multi"):
        extra_speakers = list(preset_voices.values())
    elif preset_voices:
        # Include preset voices (e.g. blessing and michael) so perturbation generalizes across genders
        extra_speakers = [v for k, v in preset_voices.items() if v != temp_speaker_path]

    try:
        result = correction_pipeline.correct(
            text=text,
            target_word=target_word,
            ref_audio_path=temp_ref_path,
            speaker_wav=temp_speaker_path,
            language=language,
            user_ref_text=ref_text,
            occurrence_index=occurrence_index,
            phonetic_hint=phonetic_hint,
            extra_speaker_wavs=extra_speakers,
        )

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)

        # Auto-persist memory to disk
        try:
            correction_pipeline.memory.save(MEMORY_PATH)
        except Exception as e:
            logger.warning(f"Could not auto-save memory to {MEMORY_PATH}: {e}")

        return JSONResponse({
            "success": True,
            "word": result.word,
            "phonetic_text": getattr(result.alignment, "auto_phonetic_hint", phonetic_hint),
            "wall_clock_seconds": result.wall_clock_seconds,
            "final_loss": result.optimization.final_loss if result.optimization else None,
            "converged": result.optimization.converged if result.optimization else None,
            "memory_size": result.memory_size,
        })
    finally:
        if os.path.exists(temp_ref_path):
            os.remove(temp_ref_path)
        if is_temp_speaker and os.path.exists(temp_speaker_path):
            os.remove(temp_speaker_path)


@app.post("/api/synthesize")
async def synthesize_text(
    text: str = Form(..., description="Text to synthesize"),
    language: str = Form("en", description="Language code"),
    speaker_wav: Optional[UploadFile] = File(None, description="Speaker reference audio for voice conditioning"),
    speaker_name: Optional[str] = Form("female", description="Preset speaker voice name if not uploading audio"),
    ref_text: Optional[str] = Form(None, description="Optional transcription of speaker audio"),
    correction_scale: Optional[float] = Form(None, description="Optional perturbation amplification scale (default 1.25)"),
):
    """Synthesize text using XTTS, automatically applying learned Hopfield corrections."""
    global inference_pipeline
    if not inference_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty.")

    temp_speaker_path, is_temp_speaker = await resolve_speaker_path(speaker_wav, speaker_name)
    output_path = tempfile.mktemp(suffix=".wav")

    try:
        synth_kwargs = {}
        if correction_scale is not None and correction_scale > 0:
            synth_kwargs["correction_scale"] = correction_scale

        result = inference_pipeline.synthesize(
            text=text,
            speaker_wav=temp_speaker_path,
            language=language,
            user_ref_text=ref_text,
            output_path=output_path,
            **synth_kwargs,
        )

        applied_spells = [f"{c['word']}->{c['spell_as']}" for c in result.get("spelling_applied", [])]
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename="synthesized_flowedit.wav",
            headers={
                "X-FlowEdit-Mode": "corrected",
                "X-Memory-Active": str(result.get("is_modified", False)),
                "X-Spelling-Applied": ", ".join(applied_spells) if applied_spells else "none",
                "X-Hopfield-Active": str(result.get("hopfield_active", False)),
            },
        )
    finally:
        if is_temp_speaker and os.path.exists(temp_speaker_path):
            os.remove(temp_speaker_path)


@app.get("/api/s3/config")
async def get_s3_config():
    """Get current S3 storage status."""
    return {
        "configured": bool(s3_spelling_store.bucket_name),
        "bucket_name": s3_spelling_store.bucket_name,
        "endpoint_url": s3_spelling_store.endpoint_url,
        "prefix": s3_spelling_store.prefix,
        "region_name": s3_spelling_store.region_name,
        "total_words": len(s3_spelling_store._dictionary),
    }


@app.post("/api/s3/config")
async def update_s3_config(
    bucket_name: Optional[str] = Form(None),
    endpoint_url: Optional[str] = Form(None),
    prefix: Optional[str] = Form(None),
    region_name: Optional[str] = Form(None),
):
    """Configure or update S3 bucket link."""
    cfg = s3_spelling_store.configure_s3(
        bucket_name=bucket_name,
        endpoint_url=endpoint_url,
        prefix=prefix,
        region_name=region_name,
    )
    return {"success": True, "config": cfg}


# ============================================================================
# S3 SPELING DICTIONARY INSPECTION & DELETION ENDPOINTS
# ============================================================================

@app.get("/api/s3/entries")
async def get_s3_entries(
    word: Optional[str] = Query(None, description="Optional target word to inspect or filter"),
    refresh: bool = Query(False, description="Re-download the latest state from S3 bucket before returning"),
):
    """View dictionary entries stored in the S3 bucket with live S3 status metadata."""
    data = s3_spelling_store.get_entries_data(refresh=refresh, word=word)
    return {
        "success": True,
        **data,
    }


@app.delete("/api/s3/entries/{word}")
async def delete_s3_entry(word: str):
    """Delete a phonetic spelling correction entry from the S3 bucket."""
    found, sync_res = s3_spelling_store.delete_word(word)
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"No spelling entry found for word '{word}' in S3 bucket '{s3_spelling_store.bucket_name}'.",
        )
    return {
        "success": True,
        "message": f"Successfully deleted '{word}' from S3 bucket '{s3_spelling_store.bucket_name}'.",
        "deleted_word": word,
        "remaining_words": len(s3_spelling_store._dictionary),
        "s3_synced": sync_res.get("synced", False),
        "s3_uri": sync_res.get("s3_uri"),
        "s3_status": sync_res.get("status"),
    }


@app.delete("/api/s3/entries/{word}/senses/{sense_id}")
async def delete_s3_entry_sense(word: str, sense_id: str):
    """Delete a specific grammatical sense of a word from the S3 bucket."""
    found, sync_res = s3_spelling_store.delete_sense(word, sense_id)
    if not found:
        raise HTTPException(
            status_code=404,
            detail=sync_res.get("error", f"Sense '{sense_id}' for word '{word}' not found in S3 dictionary."),
        )
    return {
        "success": True,
        "message": f"Deleted sense '{sense_id}' for word '{word}' from S3.",
        "word": word,
        "sense_id": sense_id,
        "remaining_words": len(s3_spelling_store._dictionary),
        "s3_synced": sync_res.get("synced", False),
        "s3_uri": sync_res.get("s3_uri"),
    }


@app.delete("/api/s3/entries")
async def clear_s3_entries(
    delete_file: bool = Query(False, description="If True, deletes the remote S3 object completely. If False, empties dictionary to {}."),
):
    """Clear or delete all entries from the S3 bucket."""
    res = s3_spelling_store.clear_all(delete_remote_file=delete_file)
    return {
        "success": True,
        "message": f"Cleared all spelling entries from S3 bucket '{s3_spelling_store.bucket_name}'.",
        "remaining_words": 0,
        "s3_synced": res.get("synced", False),
        "s3_status": res.get("status"),
        "s3_uri": res.get("s3_uri"),
        "delete_remote_file": delete_file,
    }


# Backwards compatibility aliases
@app.get("/api/spelling")
async def list_spelling_corrections(
    refresh: bool = Query(False, description="Re-download latest state from S3 before listing"),
):
    """List all registered S3 phonetic spelling corrections."""
    data = s3_spelling_store.get_entries_data(refresh=refresh)
    return {
        "size": data["total_words"],
        "corrections": data["entries"],
        "s3_bucket": data["bucket"],
        "s3_uri": data["s3_uri"],
        "s3_metadata": data["s3_metadata"],
    }


@app.delete("/api/spelling/{word}")
async def delete_spelling_correction(word: str):
    """Delete a phonetic spelling correction by word (alias for /api/s3/entries/{word})."""
    found, sync_res = s3_spelling_store.delete_word(word)
    if not found:
        raise HTTPException(status_code=404, detail=f"No spelling entry found for word '{word}'.")
    return {
        "success": True,
        "message": f"Deleted spelling correction for '{word}'.",
        "size": len(s3_spelling_store._dictionary),
        "s3_synced": sync_res.get("synced", False),
        "s3_uri": sync_res.get("s3_uri"),
    }


@app.post("/api/baseline")
async def synthesize_baseline(
    text: str = Form(..., description="Text to synthesize"),
    language: str = Form("en", description="Language code"),
    speaker_wav: Optional[UploadFile] = File(None, description="Speaker reference audio for voice conditioning"),
    speaker_name: Optional[str] = Form("female", description="Preset speaker voice name if not uploading audio"),
    ref_text: Optional[str] = Form(None, description="Optional transcription of speaker audio"),
):
    """Pure baseline synthesis without any Hopfield memory modifications (uncorrected base pronunciation)."""
    global correction_pipeline
    if not correction_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty.")

    temp_speaker_path, is_temp_speaker = await resolve_speaker_path(speaker_wav, speaker_name)
    output_path = tempfile.mktemp(suffix=".wav")

    try:
        bb = correction_pipeline.backbone
        speaker_cond = bb.get_speaker_embedding(temp_speaker_path, language, ref_text=ref_text)
        # Synthesize pure baseline using base model without FlowEdit memory
        if hasattr(bb, "synthesize_baseline"):
            wav, sr = bb.synthesize_baseline(
                text=text,
                speaker_conditioning=speaker_cond,
                language=language,
                user_ref_text=ref_text,
            )
        else:
            wav, sr = bb.synthesize_direct(
                text=text,
                speaker_conditioning=speaker_cond,
                language=language,
                user_ref_text=ref_text,
                text_embedding_delta=None,
            )
        write_audio_file(output_path, wav.squeeze().cpu().numpy(), sr)

        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename="synthesized_baseline.wav",
            headers={
                "X-FlowEdit-Mode": "baseline",
                "X-Pronunciation-Status": "uncorrected",
            },
        )
    finally:
        if is_temp_speaker and os.path.exists(temp_speaker_path):
            os.remove(temp_speaker_path)


@app.get("/api/memory")
async def get_memory_entries():
    """List all stored Hopfield memory corrections with contextual metadata."""
    global correction_pipeline
    if not correction_pipeline or not correction_pipeline.memory:
        return {"corrections": [], "size": 0}
    
    entries_info = [
        {
            "word": e.word,
            "carrier": e.carrier_text,
            "phonetic_text": getattr(e, "phonetic_text", None),
            "contexts": getattr(e, "carrier_texts", [e.carrier_text] if e.carrier_text else []),
            "access_count": e.access_count,
            "is_averaged": (e.access_count > 1),
            "language": e.language,
        }
        for e in correction_pipeline.memory.entries
    ]
    return {"size": len(entries_info), "corrections": entries_info}


@app.delete("/api/memory/{word}")
async def delete_memory_entry(word: str):
    """Delete a single Hopfield memory correction by word."""
    global correction_pipeline
    if not correction_pipeline or not correction_pipeline.memory:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    deleted = correction_pipeline.memory.delete_entry(word)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No memory entry found for word '{word}'.")

    # Auto-persist after deletion
    try:
        correction_pipeline.memory.save(MEMORY_PATH)
    except Exception as e:
        logger.warning(f"Could not auto-save memory after deletion: {e}")

    return {
        "success": True,
        "message": f"Deleted correction for '{word}'.",
        "size": correction_pipeline.memory.num_entries,
    }


@app.post("/api/memory/clear")
async def clear_memory():
    """Clear all stored Hopfield memory corrections."""
    global correction_pipeline
    if not correction_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    if correction_pipeline.memory:
        correction_pipeline.memory.clear()
        if os.path.exists(MEMORY_PATH):
            os.remove(MEMORY_PATH)
    return {"success": True, "message": "Memory cleared successfully.", "size": 0}


standalone_aligner: Optional[WhisperAligner] = None


def get_whisper_aligner() -> WhisperAligner:
    """Get active WhisperAligner instance from pipeline or initialize standalone."""
    global correction_pipeline, standalone_aligner
    if correction_pipeline and correction_pipeline.aligner:
        return correction_pipeline.aligner
    if standalone_aligner is not None:
        return standalone_aligner

    config = FlowEditConfig()
    whisper_model_env = os.environ.get("FLOWEDIT_WHISPER_MODEL", "")
    if whisper_model_env:
        config.alignment.whisper_model = whisper_model_env
    standalone_aligner = WhisperAligner(config.alignment)
    standalone_aligner.load_model()
    return standalone_aligner


@app.post("/api/transcribe")
async def transcribe_speech(
    audio: UploadFile = File(..., description="Audio file to transcribe (WAV, MP3, M4A, OGG, WebM, FLAC)"),
    language: Optional[str] = Form(None, description="Language code (e.g. 'en', 'hi', 'fr') or leave empty for auto-detection"),
    include_timestamps: bool = Form(True, description="Whether to return segment-level timestamps"),
    use_memory: bool = Form(True, description="Apply Hopfield Memory associative correction and vocabulary biasing"),
):
    """Speech-to-Text: Transcribe spoken audio to text using Whisper with Hopfield Memory spelling correction."""
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file uploaded.")

    suffix = os.path.splitext(audio.filename)[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio_path = temp_audio.name

    converted_wav_path = None
    try:
        content = await audio.read()
        if not content or len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        with open(temp_audio_path, "wb") as f:
            f.write(content)

        # Convert to strict 24kHz/16kHz Mono WAV if possible for maximum compatibility
        converted_wav_path = convert_to_wav(temp_audio_path)
        actual_path = converted_wav_path if (converted_wav_path and os.path.exists(converted_wav_path)) else temp_audio_path

        # Check if Hopfield Memory is available and has entries for vocabulary biasing
        memory_instance = None
        initial_prompt = None
        if use_memory and correction_pipeline and correction_pipeline.memory and correction_pipeline.memory.num_entries > 0:
            memory_instance = correction_pipeline.memory
            initial_prompt = memory_instance.get_vocabulary_prompt()
            if initial_prompt:
                logger.info(f"Biasing Whisper ASR with Hopfield vocabulary prompt: '{initial_prompt}'")

        aligner = get_whisper_aligner()
        result = aligner.transcribe(
            audio_path=actual_path,
            language=language,
            return_timestamps=include_timestamps,
            initial_prompt=initial_prompt,
        )

        raw_text = result["text"]
        final_text = raw_text
        corrections = []

        # Apply associative Hopfield post-correction to fix phonetic spellings to canonical words
        if memory_instance:
            final_text, corrections = memory_instance.correct_transcript(raw_text)
            if corrections:
                logger.info(f"✓ Hopfield Memory corrected {len(corrections)} spelling(s) in transcript: {corrections}")

        # Update segment texts if corrections were made
        segments = result.get("segments", [])
        if memory_instance and corrections and segments:
            updated_segments = []
            for seg in segments:
                seg_text = seg.get("text", "")
                if seg_text:
                    corr_seg_text, _ = memory_instance.correct_transcript(seg_text)
                    seg_copy = dict(seg)
                    seg_copy["text"] = corr_seg_text
                    updated_segments.append(seg_copy)
                else:
                    updated_segments.append(seg)
            segments = updated_segments

        return JSONResponse({
            "success": True,
            "text": final_text,
            "raw_text": raw_text,
            "language": result["language"],
            "duration": result["duration"],
            "segments": segments,
            "memory_applied": bool(memory_instance and corrections),
            "memory_entries_count": memory_instance.num_entries if memory_instance else 0,
            "corrections": corrections,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if converted_wav_path and os.path.exists(converted_wav_path) and converted_wav_path != temp_audio_path:
            os.remove(converted_wav_path)


@app.post("/api/transcribe/stream")
@app.post("/api/transcribe-stream", include_in_schema=False)
async def transcribe_speech_stream(
    audio: UploadFile = File(..., description="Audio file to transcribe (WAV, MP3, M4A, OGG, WebM, FLAC)"),
    language: Optional[str] = Form(None, description="Language code (e.g. 'en', 'hi', 'fr') or leave empty for auto-detection"),
    include_timestamps: bool = Form(True, description="Whether to return segment-level timestamps"),
    use_memory: bool = Form(True, description="Apply Hopfield Memory associative correction and vocabulary biasing"),
    emit_words: bool = Form(True, description="Whether to stream individual word-level events ('type': 'word')"),
):
    """Speech-to-Text Stream: Stream transcription events (Server-Sent Events) in real-time as Whisper decodes with Hopfield Memory."""
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file uploaded.")

    suffix = os.path.splitext(audio.filename)[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio_path = temp_audio.name

    content = await audio.read()
    if not content or len(content) == 0:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    with open(temp_audio_path, "wb") as f:
        f.write(content)

    converted_wav_path = convert_to_wav(temp_audio_path)
    actual_path = converted_wav_path if (converted_wav_path and os.path.exists(converted_wav_path)) else temp_audio_path

    # Check Hopfield Memory vocabulary biasing
    memory_instance = None
    initial_prompt = None
    if use_memory and correction_pipeline and correction_pipeline.memory and correction_pipeline.memory.num_entries > 0:
        memory_instance = correction_pipeline.memory
        initial_prompt = memory_instance.get_vocabulary_prompt()

    aligner = get_whisper_aligner()

    async def sse_generator():
        try:
            accumulated_raw = []
            accumulated_corrected = []
            accumulated_words = []
            all_corrections = []
            global_word_id = 0

            for event in aligner.transcribe_stream(
                audio_path=actual_path,
                language=language,
                initial_prompt=initial_prompt,
            ):
                event_type = event.get("type")
                if event_type == "metadata":
                    yield f"data: {json.dumps(event)}\n\n"
                elif event_type == "segment":
                    seg_raw = event.get("text", "")
                    seg_corrected = seg_raw
                    seg_corrections = []
                    if memory_instance and seg_raw:
                        seg_corrected, seg_corrections = memory_instance.correct_transcript(seg_raw)
                        all_corrections.extend(seg_corrections)

                    accumulated_raw.append(seg_raw)
                    accumulated_corrected.append(seg_corrected)

                    seg_start = event.get("start", 0.0)
                    seg_end = event.get("end", 0.0)

                    # Stream word-by-word events in real-time
                    if emit_words and seg_corrected:
                        words = seg_corrected.split()
                        w_dur = max(seg_end - seg_start, 0.05)
                        w_step = w_dur / max(len(words), 1)
                        for w_i, word_text in enumerate(words):
                            accumulated_words.append(word_text)
                            w_start = round(seg_start + w_i * w_step, 3)
                            w_end = round(seg_start + (w_i + 1) * w_step, 3)
                            word_event = {
                                "type": "word",
                                "id": global_word_id,
                                "start": w_start,
                                "end": w_end,
                                "word": word_text,
                                "partial_transcript": " ".join(accumulated_words),
                            }
                            global_word_id += 1
                            yield f"data: {json.dumps(word_event)}\n\n"
                            await asyncio.sleep(0)

                    out_event = {
                        "type": "segment",
                        "id": event.get("id", 0),
                        "start": seg_start,
                        "end": seg_end,
                        "text": seg_corrected,
                        "raw_text": seg_raw,
                        "partial_transcript": " ".join(accumulated_corrected),
                        "corrections": seg_corrections,
                    }
                    yield f"data: {json.dumps(out_event)}\n\n"
                    await asyncio.sleep(0)
                elif event_type == "complete":
                    full_raw = event.get("text", "")
                    full_final = full_raw
                    full_corrections = []
                    if memory_instance and full_raw:
                        full_final, full_corrections = memory_instance.correct_transcript(full_raw)

                    # Update segments with corrected text
                    raw_segments = event.get("segments", [])
                    final_segments = []
                    for seg in raw_segments:
                        st = seg.get("text", "")
                        if st and memory_instance:
                            corr_st, _ = memory_instance.correct_transcript(st)
                            sc = dict(seg)
                            sc["text"] = corr_st
                            final_segments.append(sc)
                        else:
                            final_segments.append(seg)

                    complete_event = {
                        "type": "complete",
                        "success": True,
                        "text": full_final,
                        "raw_text": full_raw,
                        "language": event.get("language", language or "unknown"),
                        "duration": event.get("duration", 0.0),
                        "segments": final_segments,
                        "memory_applied": bool(memory_instance and (full_corrections or all_corrections)),
                        "corrections": full_corrections or all_corrections,
                    }
                    yield f"data: {json.dumps(complete_event)}\n\n"
        except Exception as e:
            logger.error(f"Streaming transcription error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            if os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except Exception:
                    pass
            if converted_wav_path and os.path.exists(converted_wav_path) and converted_wav_path != temp_audio_path:
                try:
                    os.remove(converted_wav_path)
                except Exception:
                    pass

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

