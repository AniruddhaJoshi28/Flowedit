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
    pipeline = CorrectionLoop(config)
    pipeline.load_models(memory_path=MEMORY_PATH)
    print("Models loaded successfully.")
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
    Learn a pronunciation correction from reference audio.
    """
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet.")

    # Save uploaded files to temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref:
        shutil.copyfileobj(ref_audio.file, temp_ref)
        temp_ref_path = temp_ref.name

    temp_speaker_path = None
    if speaker_wav:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
            shutil.copyfileobj(speaker_wav.file, temp_speaker)
            temp_speaker_path = temp_speaker.name

    try:
        # Run correction pipeline
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

@app.post("/api/synthesize")
async def synthesize_text(
    text: str = Form(..., description="Text to synthesize"),
    language: str = Form("en", description="Language code"),
    speaker_wav: UploadFile = File(..., description="Speaker reference audio for voice conditioning"),
    ref_text: Optional[str] = Form(None, description="Optional transcription of the speaker audio. If empty, Whisper will auto-transcribe."),
):
    """
    Synthesize text, automatically applying learned corrections.
    """
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
        shutil.copyfileobj(speaker_wav.file, temp_speaker)
        temp_speaker_path = temp_speaker.name

    output_path = tempfile.mktemp(suffix=".wav")

    try:
        # Step 1: Get text embeddings
        base_embeddings = pipeline.backbone.encode_text(text, language)
        
        # Step 2: Retrieve corrections and apply gating via the Hopfield Refiner
        from flowedit.refiner.hopfield_refiner import HopfieldRefiner
        refiner = HopfieldRefiner(pipeline.memory, config=pipeline.config.memory).to(pipeline.backbone.device)
        corrected_embeddings, gate_values = refiner(base_embeddings, text=text)
        
        corrections_applied = (gate_values > 0.5).sum().item()
        max_gate = gate_values.max().item() if gate_values.numel() > 0 else 0.0
        diff_norm = torch.norm(corrected_embeddings - base_embeddings).item()
        
        print(f"[Synthesize] Memory size: {pipeline.memory.size}, "
              f"Corrections applied: {corrections_applied}, "
              f"Max gate: {max_gate:.4f}, "
              f"Embedding diff norm: {diff_norm:.4f}")
        
        # Step 3: Get speaker conditioning
        speaker_conditioning = pipeline.backbone.get_speaker_embedding(temp_speaker_path, language, ref_text=ref_text)
        
        # Step 4: Synthesize
        # Use direct synthesis (no embedding hooks) when corrections don't
        # actually change the embeddings.  The hook in synthesize_from_embeddings
        # replaces context-aware embeddings with context-free ones, which
        # causes extra words / repetition in the output audio.
        if diff_norm < 1e-4:
            print("[Synthesize] No meaningful embedding changes → using direct F5-TTS synthesis (no hooks)")
            waveform, sr = pipeline.backbone.synthesize_direct(
                text=text,
                speaker_conditioning=speaker_conditioning,
                language=language,
                user_ref_text=ref_text,
            )
        else:
            print(f"[Synthesize] Corrections active (diff={diff_norm:.4f}) → using hook-based synthesis")
            waveform, sr = pipeline.backbone.synthesize_from_embeddings(
                text_embeddings=corrected_embeddings.detach(),
                speaker_conditioning=speaker_conditioning,
                text=text,
                language=language,
            )
        
        # Save waveform to output_path using soundfile
        import soundfile as sf
        sf.write(output_path, waveform.squeeze().detach().cpu().numpy(), sr)

        return FileResponse(
            path=output_path, 
            media_type="audio/wav", 
            filename="synthesized.wav",
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
