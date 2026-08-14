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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from flowedit.config import FlowEditConfig
from flowedit.audio.prompt_validator import ReferenceAudioError
from flowedit.pipeline.correction_loop import CorrectionLoop
from flowedit.pipeline.inference import FlowEditInference

logger = logging.getLogger(__name__)

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
    config.backbone.f5tts_ckpt_file = os.environ.get("FLOWEDIT_F5TTS_CKPT", "")
    config.backbone.f5tts_vocab_file = os.environ.get("FLOWEDIT_F5TTS_VOCAB", "")
    config.backbone.vocoder_local_path = os.environ.get("FLOWEDIT_VOCODER_DIR", "")
    whisper_model_env = os.environ.get("FLOWEDIT_WHISPER_MODEL", "")
    if whisper_model_env:
        config.alignment.whisper_model = whisper_model_env

    correction_pipeline = CorrectionLoop(config)
    correction_pipeline.load_models()

    inference_pipeline = FlowEditInference(config)
    inference_pipeline.load(
        backbone=correction_pipeline.backbone,
        memory=correction_pipeline.memory,
    )

    logger.info("FlowEdit API ready.")
    yield
    logger.info("Shutting down FlowEdit API...")


app = FastAPI(
    title="FlowEdit API",
    description="Lifelong Pronunciation Adaptation for Flow-Matching TTS via Associative Memory (arXiv:2606.20518)",
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


@app.get("/")
def read_root():
    return {"message": "FlowEdit API (arXiv:2606.20518) is active. Visit /docs for OpenAPI specs."}


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
    speaker_wav: UploadFile = File(..., description="Speaker reference audio for voice conditioning"),
    ref_text: Optional[str] = Form(None, description="Optional transcription of speaker audio"),
    occurrence_index: int = Form(0, description="Occurrence index of target word if multiple exist"),
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

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
        temp_speaker_path = temp_speaker.name
    await speaker_wav.seek(0)
    with open(temp_speaker_path, "wb") as f:
        f.write(await speaker_wav.read())
    temp_speaker_path = convert_to_wav(temp_speaker_path)

    try:
        result = correction_pipeline.correct(
            text=text,
            target_word=target_word,
            ref_audio_path=temp_ref_path,
            speaker_wav=temp_speaker_path,
            language=language,
            user_ref_text=ref_text,
            occurrence_index=occurrence_index,
        )

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)

        return JSONResponse({
            "success": True,
            "word": result.word,
            "wall_clock_seconds": result.wall_clock_seconds,
            "final_loss": result.optimization.final_loss if result.optimization else None,
            "converged": result.optimization.converged if result.optimization else None,
            "memory_size": result.memory_size,
        })
    finally:
        if os.path.exists(temp_ref_path):
            os.remove(temp_ref_path)
        if os.path.exists(temp_speaker_path):
            os.remove(temp_speaker_path)


@app.post("/api/synthesize")
async def synthesize_text(
    text: str = Form(..., description="Text to synthesize"),
    language: str = Form("en", description="Language code"),
    speaker_wav: UploadFile = File(..., description="Speaker reference audio for voice conditioning"),
    ref_text: Optional[str] = Form(None, description="Optional transcription of speaker audio"),
):
    """Synthesize text using F5-TTS, automatically applying learned Hopfield corrections."""
    global inference_pipeline
    if not inference_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
        temp_speaker_path = temp_speaker.name
    await speaker_wav.seek(0)
    with open(temp_speaker_path, "wb") as f:
        f.write(await speaker_wav.read())
    temp_speaker_path = convert_to_wav(temp_speaker_path)

    output_path = tempfile.mktemp(suffix=".wav")

    try:
        result = inference_pipeline.synthesize(
            text=text,
            speaker_wav=temp_speaker_path,
            language=language,
            user_ref_text=ref_text,
            output_path=output_path,
        )

        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename="synthesized_flowedit.wav",
        )
    finally:
        if os.path.exists(temp_speaker_path):
            os.remove(temp_speaker_path)


@app.post("/api/baseline")
async def synthesize_baseline(
    text: str = Form(..., description="Text to synthesize"),
    language: str = Form("en", description="Language code"),
    speaker_wav: UploadFile = File(..., description="Speaker reference audio for voice conditioning"),
    ref_text: Optional[str] = Form(None, description="Optional transcription of speaker audio"),
):
    """Pure baseline synthesis without any Hopfield memory modifications."""
    global correction_pipeline
    if not correction_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    text = text.strip()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
        temp_speaker_path = temp_speaker.name
    await speaker_wav.seek(0)
    with open(temp_speaker_path, "wb") as f:
        f.write(await speaker_wav.read())
    temp_speaker_path = convert_to_wav(temp_speaker_path)

    output_path = tempfile.mktemp(suffix=".wav")

    try:
        bb = correction_pipeline.backbone
        speaker_cond = bb.get_speaker_embedding(temp_speaker_path, language, ref_text=ref_text)
        wav, sr = bb.synthesize_direct(
            text=text,
            speaker_conditioning=speaker_cond,
            language=language,
            user_ref_text=ref_text,
        )
        sf.write(output_path, wav.squeeze().cpu().numpy(), sr)

        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename="synthesized_baseline.wav",
        )
    finally:
        if os.path.exists(temp_speaker_path):
            os.remove(temp_speaker_path)


@app.get("/api/memory")
async def get_memory_entries():
    """List all stored Hopfield memory corrections."""
    global correction_pipeline
    if not correction_pipeline or not correction_pipeline.memory:
        return {"corrections": [], "size": 0}
    
    entries_info = [
        {"word": e.word, "carrier": e.carrier_text, "access_count": e.access_count}
        for e in correction_pipeline.memory.entries
    ]
    return {"size": len(entries_info), "corrections": entries_info}


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
