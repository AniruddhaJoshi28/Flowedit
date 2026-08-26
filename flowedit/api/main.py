"""
FlowEdit FastAPI REST Service.

Provides endpoints for:
- /api/correct: Learn a pronunciation correction from reference audio
- /api/synthesize: Synthesize speech with automatic Hopfield memory retrieval
- /api/baseline: Vanilla baseline synthesis without memory
- /api/memory: Inspect or clear stored associative memory
"""

import os
import shutil
import tempfile
import subprocess
import traceback
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from flowedit.config import FlowEditConfig
from flowedit.audio.prompt_validator import ReferenceAudioError
from flowedit.pipeline.correction_loop import CorrectionLoop
from flowedit.pipeline.inference import FlowEditInference

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
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(input_path)
        return output_path
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
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


def get_preset_voices() -> Dict[str, str]:
    """Find preset deployment voice WAV files."""
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "text_to_speech", "app", "deploy_voices")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "deploy_voices")),
        "/home/rsurya/projects/text_to_speech/app/deploy_voices",
        os.path.expanduser("~/projects/text_to_speech/app/deploy_voices"),
    ]
    voices = {}
    for d in candidates:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".wav"):
                    name = os.path.splitext(f)[0].lower()
                    voices[name] = os.path.join(d, f)
            if voices:
                break
    return voices


async def resolve_speaker_path(speaker_wav: Optional[UploadFile], speaker_name: Optional[str]) -> tuple[str, bool]:
    """Resolves speaker WAV file path from upload or preset name. Returns (path, is_temp)."""
    if speaker_wav is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
            temp_path = temp_speaker.name
        content = await speaker_wav.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        converted_path = convert_to_wav(temp_path)
        return converted_path, True
    
    # Check preset
    voices = get_preset_voices()
    name_key = (speaker_name or "female").lower()
    if name_key in ("female", "woman", "blessing"):
        target_name = "blessing" if "blessing" in voices else next(iter(voices.keys()), None)
    elif name_key in ("male", "man", "michael"):
        target_name = "michael" if "michael" in voices else next(iter(voices.keys()), None)
    else:
        target_name = name_key if name_key in voices else next(iter(voices.keys()), None)
    
    if target_name and target_name in voices and os.path.isfile(voices[target_name]):
        return voices[target_name], False
    
    # Fallback to any wav file found
    if voices:
        first_voice = next(iter(voices.values()))
        return first_voice, False

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
    ref_audio: UploadFile = File(..., description="Reference audio with correct pronunciation"),
    speaker_wav: Optional[UploadFile] = File(None, description="Speaker reference audio for voice conditioning"),
    speaker_name: Optional[str] = Form("female", description="Preset speaker voice name if not uploading audio"),
    ref_text: Optional[str] = Form(None, description="Optional transcription of speaker audio"),
    occurrence_index: int = Form(0, description="Occurrence index of target word if multiple exist"),
    phonetic_hint: Optional[str] = Form(None, description="Phonetic hint for warm-starting optimization"),
):
    """Learn a pronunciation correction from reference audio using FlowEdit (Stages 1-3)."""
    global correction_pipeline
    if not correction_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    text = text.strip()
    target_word = target_word.strip()

    if not text or not target_word:
        raise HTTPException(status_code=400, detail="text and target_word must be non-empty.")

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

        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename="synthesized_flowedit.wav",
            headers={
                "X-FlowEdit-Mode": "corrected",
                "X-Memory-Active": str(result.get("is_modified", False)),
            },
        )
    finally:
        if is_temp_speaker and os.path.exists(temp_speaker_path):
            os.remove(temp_speaker_path)


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
        sf.write(output_path, wav.squeeze().cpu().numpy(), sr)

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
