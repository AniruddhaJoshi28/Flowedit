import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Optional
import torch

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
from fastapi import Request

from flowedit.config import FlowEditConfig
from flowedit.audio.prompt_validator import ReferenceAudioError
# Workaround for CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on Linux servers

torch.backends.cudnn.enabled = False

from flowedit.pipeline.correction_loop import CorrectionLoop

# Global pipeline instance
pipeline = None
MEMORY_PATH = "./corrections.pt"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global pipeline
    print("Loading FlowEdit pipeline models...")
    config = FlowEditConfig()
    backbone_env = os.environ.get("FLOWEDIT_BACKBONE", "f5tts")
    config.backbone.backbone_type = backbone_env.lower()
    config.backbone.f5tts_ckpt_file = os.environ.get("FLOWEDIT_F5TTS_CKPT", "")
    config.backbone.f5tts_vocab_file = os.environ.get("FLOWEDIT_F5TTS_VOCAB", "")
    config.backbone.vocoder_local_path = os.environ.get("FLOWEDIT_VOCODER_DIR", "")
    whisper_model_env = os.environ.get("FLOWEDIT_WHISPER_MODEL", "")
    if whisper_model_env:
        config.alignment.whisper_model = whisper_model_env
    print(f"  Whisper model path: {config.alignment.whisper_model}")
    print(f"  Backbone type: {config.backbone.backbone_type.upper()}")

    if config.backbone.backbone_type == "xtts":
        print(f"  XTTS model dir: {config.backbone.xtts_model_dir}")
        print(f"  XTTS checkpoint: {config.backbone.xtts_checkpoint}")
    pipeline = CorrectionLoop(config)
    pipeline.load_models(memory_path=MEMORY_PATH)
    app.state.pipeline = pipeline
    backbone_cls = type(pipeline.backbone).__name__
    print(f"Models loaded successfully. Active backbone: {backbone_cls}")
    yield
    # Shutdown logic
    print("Shutting down FlowEdit API...")

app = FastAPI(
    title="FlowEdit API",
    description="API for the FlowEdit Pronunciation Adaptation Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Swagger UI / Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "FlowEdit API is running. Visit /docs for Swagger UI."}

@app.exception_handler(ReferenceAudioError)
async def reference_audio_exception_handler(request: Request, exc: ReferenceAudioError):
    return JSONResponse(
        status_code=422,
        content={
            "code": getattr(exc, "code", "REFERENCE_AUDIO_INVALID"),
            "message": str(exc),
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": tb}
    )


@app.post("/api/correct")
async def correct_pronunciation(
    text: str = Form(..., description="Full text containing the target word"),
    target_word: str = Form(..., description="The word to correct pronunciation of"),
    language: str = Form("en", description="Language code"),
    ref_audio: UploadFile = File(..., description="Reference audio with correct pronunciation"),
    speaker_wav: UploadFile | None = None,
):
    """
    Learn a pronunciation correction from reference audio using the F5-TTS model backbone.
    """
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet.")

    text = text.strip()
    target_word = target_word.strip()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref:
        temp_ref_path = temp_ref.name
        
    await ref_audio.seek(0)
    with open(temp_ref_path, "wb") as f:
        f.write(await ref_audio.read())
        
    temp_ref_path = convert_to_wav(temp_ref_path)

    temp_speaker_path = None
    if speaker_wav:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
            temp_speaker_path = temp_speaker.name
            
        await speaker_wav.seek(0)
        with open(temp_speaker_path, "wb") as f:
            f.write(await speaker_wav.read())
            
        temp_speaker_path = convert_to_wav(temp_speaker_path)

    try:
        # Run correction pipeline with requested backbone
        result = pipeline.correct(
            text=text,
            target_word=target_word,
            ref_audio_path=temp_ref_path,
            speaker_wav=temp_speaker_path,
            language=language,
        )

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)

        # Save memory
        pipeline.save_memory(MEMORY_PATH)

        return JSONResponse({
            "success": True,
            "word": result.word,
            "backbone": "f5tts",
            "wall_clock_seconds": result.wall_clock_seconds,
            "final_loss": result.optimization.final_loss if hasattr(result, "optimization") and hasattr(result.optimization, "final_loss") else None,
            "converged": result.optimization.converged if hasattr(result, "optimization") and hasattr(result.optimization, "converged") else None,
            "memory_size": result.memory_size,
        })
    finally:
        # Cleanup temp files
        if os.path.exists(temp_ref_path):
            os.remove(temp_ref_path)
        if temp_speaker_path and os.path.exists(temp_speaker_path):
            os.remove(temp_speaker_path)

import subprocess

def convert_to_wav(input_path: str) -> str:
    """Uses system ffmpeg to ensure any uploaded audio file is a strict 24kHz Mono WAV."""
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

@app.post("/api/synthesize")
async def synthesize_text(
    text: str = Form(..., description="Text to synthesize"),
    language: str = Form("en", description="Language code"),
    speaker_wav: UploadFile = File(..., description="Speaker reference audio for voice conditioning"),
    ref_text: Optional[str] = Form(None, description="Optional transcription of the speaker audio."),
):
    """
    Synthesize text using the F5-TTS backbone model, automatically applying learned Hopfield corrections.
    """
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet.")

    text = text.strip()
    if ref_text is not None:
        ref_text = ref_text.strip()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
        temp_speaker_path = temp_speaker.name
        
    await speaker_wav.seek(0)
    with open(temp_speaker_path, "wb") as f:
        f.write(await speaker_wav.read())
        
    # Auto-convert uploaded file to strict WAV format
    temp_speaker_path = convert_to_wav(temp_speaker_path)

    output_path = tempfile.mktemp(suffix=".wav")

    try:
        # Get requested backbone
        bb = pipeline.get_backbone()

        # Step 1: Get text embeddings
        base_embeddings = bb.encode_text(text, language)
        
        # Step 2: Retrieve corrections and apply gating via the Hopfield Refiner
        from flowedit.refiner.hopfield_refiner import HopfieldRefiner
        refiner = HopfieldRefiner(pipeline.memory, config=pipeline.config.memory).to(bb.device)
        corrected_embeddings, gate_values = refiner(base_embeddings, text=text)
        
        corrections_applied = (gate_values > 0.5).sum().item()
        max_gate = gate_values.max().item() if gate_values.numel() > 0 else 0.0
        diff_norm = torch.norm(corrected_embeddings - base_embeddings).item()
        
        print(f"[Synthesize] Backbone: {pipeline.config.backbone.backbone_type.upper()}, Memory size: {pipeline.memory.size}, "
              f"Corrections applied: {corrections_applied}, "
              f"Max gate: {max_gate:.4f}, "
              f"Embedding diff norm: {diff_norm:.4f}")
        
        # Step 3: Get speaker conditioning
        speaker_conditioning = bb.get_speaker_embedding(temp_speaker_path, language, ref_text=ref_text)
        
        # Step 4: Synthesize — use hook-based synthesis whenever corrections are active
        backbone_name = type(bb).__name__
        
        # Debug: log memory delta norms
        if pipeline.memory and not pipeline.memory.is_empty:
            for i, (v, m) in enumerate(zip(pipeline.memory.values, pipeline.memory.metadata)):
                v_norm = torch.norm(v).item()
                print(f"  [Memory {i}] word='{m.get('word', '?')}', δ_norm={v_norm:.6f}")
        
        if corrections_applied > 0 and diff_norm > 1e-4:
            embedding_delta = corrected_embeddings.detach() - base_embeddings.detach()
            print(f"[Synthesize] Corrections active (applied={corrections_applied}, diff={diff_norm:.4f}) → using hook-based NORMAL inference on {backbone_name}")
            waveform, sr = bb.synthesize_direct(
                text=text,
                speaker_conditioning=speaker_conditioning,
                language=language,
                user_ref_text=ref_text,
                text_embedding_delta=embedding_delta
            )
        else:
            reason = "no corrections matched" if corrections_applied == 0 else f"diff_norm too small ({diff_norm:.6f})"
            print(f"[Synthesize] {reason} → using direct {backbone_name} synthesis")
            waveform, sr = bb.synthesize_direct(
                text=text,
                speaker_conditioning=speaker_conditioning,
                language=language,
                user_ref_text=ref_text,
            )
        
        # Save waveform to output_path using soundfile
        import soundfile as sf
        sf.write(output_path, waveform.squeeze().detach().cpu().numpy(), sr)

        return FileResponse(
            path=output_path, 
            media_type="audio/wav", 
            filename=f"synthesized_f5tts.wav",
            background=None
        )
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        import traceback
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\nTraceback:\n{tb}")
    finally:
        # Cleanup speaker temp file
        if os.path.exists(temp_speaker_path):
            os.remove(temp_speaker_path)

@app.post("/api/baseline")
async def synthesize_baseline_api(
    text: str = Form(..., description="Text to synthesize"),
    language: str = Form("en", description="Language code"),
    speaker_wav: UploadFile = File(..., description="Speaker reference audio for voice conditioning"),
    ref_text: Optional[str] = Form(None, description="Optional transcription of the speaker audio."),
):
    """
    Phase 1 Baseline Endpoint: Pure native F5-TTS synthesis without hooks or patches.
    """
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
        temp_speaker_path = temp_speaker.name
        
    await speaker_wav.seek(0)
    with open(temp_speaker_path, "wb") as f:
        f.write(await speaker_wav.read())
        
    temp_speaker_path = convert_to_wav(temp_speaker_path)
    output_path = tempfile.mktemp(suffix=".wav")

    try:
        bb = pipeline.get_backbone()
        speaker_conditioning = bb.get_speaker_embedding(temp_speaker_path, language, ref_text=ref_text)
        
        print(f"[Synthesize Baseline] Calling pure F5-TTS...")
        waveform, sr = bb.synthesize_baseline(
            text=text,
            speaker_conditioning=speaker_conditioning,
            language=language,
            user_ref_text=ref_text,
        )
        
        import soundfile as sf
        sf.write(output_path, waveform.squeeze().detach().cpu().numpy(), sr)

        return FileResponse(
            path=output_path, 
            media_type="audio/wav", 
            filename="synthesized_baseline.wav",
            background=None
        )
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        import traceback
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\nTraceback:\n{tb}")
    finally:
        if os.path.exists(temp_speaker_path):
            os.remove(temp_speaker_path)


