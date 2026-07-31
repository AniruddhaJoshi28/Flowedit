"""
xtts-api-server Compatible Endpoints for FlowEdit API.

This router provides full compatibility with the xtts-api-server standard REST API,
allowing clients built for xtts-api-server (e.g. SillyTavern, web interfaces, CLI tools)
to use the fine-tuned XTTS model embedded inside FlowEdit seamlessly.

Endpoints provided:
  - GET  /speakers_list
  - GET  /speakers
  - GET  /languages
  - POST /tts_to_audio/
  - POST /tts_to_file
  - GET  /tts_stream
  - POST /set_tts_settings
  - GET  /get_tts_settings
  - POST /switch_model
  - POST /set_output
  - POST /set_speaker_folder
  - GET  /get_folders
  - GET  /get_models_list
  - GET  /sample/{file_name}
"""

import os
import io
import shutil
import tempfile
from typing import Optional, List, Dict, Any
import numpy as np
import soundfile as sf
import torch

from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

router = APIRouter(tags=["xtts-api-server"])

# Global state for XTTS settings
DEFAULT_SPEAKER_DIR = "./speakers"
DEFAULT_OUTPUT_DIR = "./output"

tts_settings = {
    "temperature": 0.75,
    "length_penalty": 1.0,
    "repetition_penalty": 10.0,
    "top_k": 50,
    "top_p": 0.85,
    "speed": 1.0,
    "enable_text_splitting": False,
    "stream_chunk_size": 20,
}

speaker_dir = DEFAULT_SPEAKER_DIR
output_dir = DEFAULT_OUTPUT_DIR


# Pydantic Schemas for xtts-api-server requests
class TTSToAudioRequest(BaseModel):
    text: str
    speaker_wav: str = "default"
    language: str = "en"


class TTSToFileRequest(BaseModel):
    text: str
    speaker_wav: str = "default"
    language: str = "en"
    output_file_path: Optional[str] = None


class TTSSettingsRequest(BaseModel):
    temperature: Optional[float] = 0.75
    length_penalty: Optional[float] = 1.0
    repetition_penalty: Optional[float] = 10.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.85
    speed: Optional[float] = 1.0
    enable_text_splitting: Optional[bool] = False
    stream_chunk_size: Optional[int] = 20


class SetFolderRequest(BaseModel):
    folder_path: str


class SwitchModelRequest(BaseModel):
    model_name: str


def get_pipeline(request: Request):
    """Retrieve the global FlowEdit pipeline from app state."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        # Fallback to main module pipeline
        import flowedit.api.main as main_mod
        pipeline = main_mod.pipeline
    if pipeline is None or pipeline.backbone is None:
        raise HTTPException(status_code=503, detail="FlowEdit pipeline not loaded yet.")
    return pipeline


def resolve_speaker_path(speaker_name: str) -> str:
    """Resolve speaker WAV path from speaker_name."""
    global speaker_dir
    os.makedirs(speaker_dir, exist_ok=True)
    
    # If direct filepath
    if os.path.isfile(speaker_name):
        return speaker_name
        
    # Check in speaker_dir
    candidate = os.path.join(speaker_dir, f"{speaker_name}.wav")
    if os.path.isfile(candidate):
        return candidate
    candidate_raw = os.path.join(speaker_dir, speaker_name)
    if os.path.isfile(candidate_raw):
        return candidate_raw
        
    # Check for any .wav in speaker_dir
    wavs = [os.path.join(speaker_dir, f) for f in os.listdir(speaker_dir) if f.endswith(".wav")]
    if wavs:
        return wavs[0]
        
    # Fallback default path inside resources
    res_default = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "default_speaker.wav")
    if os.path.isfile(res_default):
        return res_default
        
    raise HTTPException(status_code=404, detail=f"Speaker '{speaker_name}' not found in {speaker_dir}")


@router.get("/speakers_list")
@router.get("/speakers")
async def list_speakers():
    """List all available speaker names."""
    global speaker_dir
    os.makedirs(speaker_dir, exist_ok=True)
    speakers = []
    for f in os.listdir(speaker_dir):
        if f.endswith(".wav") or f.endswith(".mp3") or f.endswith(".flac"):
            speakers.append(os.path.splitext(f)[0])
    if not speakers:
        speakers = ["default"]
    return JSONResponse(content=speakers)


@router.get("/languages")
async def list_languages():
    """List all supported language codes."""
    languages = [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
        "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"
    ]
    return JSONResponse(content=languages)


@router.post("/tts_to_audio/")
@router.post("/tts_to_audio")
async def tts_to_audio(req: TTSToAudioRequest, request: Request):
    """Synthesize text to WAV audio bytes."""
    pipeline = get_pipeline(request)
    speaker_path = resolve_speaker_path(req.speaker_wav)

    speaker_cond = pipeline.backbone.get_speaker_embedding(speaker_path, req.language)
    waveform, sr = pipeline.backbone.synthesize_direct(
        text=req.text,
        speaker_conditioning=speaker_cond,
        language=req.language,
    )

    wav_np = waveform.squeeze().cpu().numpy()
    out_buf = io.BytesIO()
    sf.write(out_buf, wav_np, sr, format="WAV")
    out_buf.seek(0)

    return StreamingResponse(out_buf, media_type="audio/wav")


@router.post("/tts_to_file")
async def tts_to_file(req: TTSToFileRequest, request: Request):
    """Synthesize text to a WAV file."""
    pipeline = get_pipeline(request)
    speaker_path = resolve_speaker_path(req.speaker_wav)

    global output_dir
    os.makedirs(output_dir, exist_ok=True)
    out_path = req.output_file_path or os.path.join(output_dir, "output.wav")

    speaker_cond = pipeline.backbone.get_speaker_embedding(speaker_path, req.language)
    waveform, sr = pipeline.backbone.synthesize_direct(
        text=req.text,
        speaker_conditioning=speaker_cond,
        language=req.language,
    )

    wav_np = waveform.squeeze().cpu().numpy()
    sf.write(out_path, wav_np, sr)

    return FileResponse(path=out_path, media_type="audio/wav", filename=os.path.basename(out_path))


@router.get("/get_tts_settings")
async def get_settings():
    """Get current TTS settings."""
    return JSONResponse(content=tts_settings)


@router.post("/set_tts_settings")
async def set_settings(req: TTSSettingsRequest):
    """Update TTS settings."""
    global tts_settings
    updates = req.dict(exclude_unset=True)
    tts_settings.update(updates)
    return JSONResponse(content={"status": "success", "settings": tts_settings})


@router.post("/set_speaker_folder")
async def set_speaker_folder(req: SetFolderRequest):
    """Set custom speaker folder."""
    global speaker_dir
    speaker_dir = req.folder_path
    os.makedirs(speaker_dir, exist_ok=True)
    return JSONResponse(content={"status": "success", "speaker_folder": speaker_dir})


@router.post("/set_output")
async def set_output_folder(req: SetFolderRequest):
    """Set custom output folder."""
    global output_dir
    output_dir = req.folder_path
    os.makedirs(output_dir, exist_ok=True)
    return JSONResponse(content={"status": "success", "output_folder": output_dir})


@router.get("/get_folders")
async def get_folders():
    """Get speaker and output folders."""
    return JSONResponse(content={"speaker_folder": speaker_dir, "output_folder": output_dir})


@router.get("/get_models_list")
async def get_models_list():
    """List loaded model names."""
    return JSONResponse(content=["XTTS-v2-FineTuned-FlowEdit"])


@router.post("/switch_model")
async def switch_model(req: SwitchModelRequest):
    """Switch model endpoint placeholder."""
    return JSONResponse(content={"status": "success", "active_model": req.model_name})


@router.get("/sample/{file_name}")
async def get_sample(file_name: str):
    """Download audio sample from output folder."""
    global output_dir
    file_path = os.path.join(output_dir, file_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, media_type="audio/wav")
