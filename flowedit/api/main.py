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

from flowedit.api.xtts_routes import router as xtts_router

# Global pipeline instance
pipeline = None
MEMORY_PATH = "./corrections.pt"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global pipeline
    print("Loading FlowEdit pipeline models...")
    config = FlowEditConfig()
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
    description="API for the FlowEdit Pronunciation Adaptation Pipeline (with xtts-api-server support)",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(xtts_router)

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
    backbone_type: str = Form("xtts", description="Backbone model to use: 'xtts' or 'f5tts'"),
    ref_audio: UploadFile = File(..., description="Reference audio with correct pronunciation"),
    speaker_wav: UploadFile | None = None,
):
    """
    Learn a pronunciation correction from reference audio using the specified model backbone.
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
        # Run correction pipeline with requested backbone
        result = pipeline.correct(
            text=text,
            target_word=target_word,
            ref_audio_path=temp_ref_path,
            speaker_wav=temp_speaker_path,
            language=language,
            backbone_type=backbone_type,
        )

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)

        # Save memory
        pipeline.save_memory(MEMORY_PATH)

        return JSONResponse({
            "success": True,
            "word": result.word,
            "backbone": backbone_type,
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
    backbone_type: str = Form("xtts", description="Backbone model to use: 'xtts' or 'f5tts'"),
    speaker_wav: UploadFile = File(..., description="Speaker reference audio for voice conditioning"),
    ref_text: Optional[str] = Form(None, description="Optional transcription of the speaker audio."),
):
    """
    Synthesize text using requested backbone model (xtts or f5tts), automatically applying learned Hopfield corrections.
    """
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
        shutil.copyfileobj(speaker_wav.file, temp_speaker)
        temp_speaker_path = temp_speaker.name

    output_path = tempfile.mktemp(suffix=".wav")

    try:
        # Get requested backbone
        bb = pipeline.get_backbone(backbone_type)

        # Step 1: Get text embeddings
        base_embeddings = bb.encode_text(text, language)
        
        # Step 2: Retrieve corrections and apply gating via the Hopfield Refiner
        from flowedit.refiner.hopfield_refiner import HopfieldRefiner
        refiner = HopfieldRefiner(pipeline.memory, config=pipeline.config.memory).to(bb.device)
        corrected_embeddings, gate_values = refiner(base_embeddings, text=text)
        
        corrections_applied = (gate_values > 0.5).sum().item()
        max_gate = gate_values.max().item() if gate_values.numel() > 0 else 0.0
        diff_norm = torch.norm(corrected_embeddings - base_embeddings).item()
        
        print(f"[Synthesize] Backbone: {backbone_type.upper()}, Memory size: {pipeline.memory.size}, "
              f"Corrections applied: {corrections_applied}, "
              f"Max gate: {max_gate:.4f}, "
              f"Embedding diff norm: {diff_norm:.4f}")
        
        # Step 3: Get speaker conditioning
        speaker_conditioning = bb.get_speaker_embedding(temp_speaker_path, language, ref_text=ref_text)
        
        # Step 4: Synthesize
        backbone_name = type(bb).__name__
        if diff_norm < 1e-4:
            print(f"[Synthesize] No meaningful embedding changes → using direct {backbone_name} synthesis")
            waveform, sr = bb.synthesize_direct(
                text=text,
                speaker_conditioning=speaker_conditioning,
                language=language,
                user_ref_text=ref_text,
            )
        else:
            print(f"[Synthesize] Corrections active (diff={diff_norm:.4f}) → using hook-based synthesis on {backbone_name}")
            waveform, sr = bb.synthesize_from_embeddings(
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
            filename=f"synthesized_{backbone_type}.wav",
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

@app.post("/api/synthesize_raw")
async def synthesize_raw(
    text: str = Form(..., description="Text to synthesize"),
    language: str = Form("en", description="Language code"),
    speaker_wav: UploadFile = File(..., description="Speaker reference audio"),
    temperature: float = Form(0.75, description="Sampling temperature"),
    repetition_penalty: float = Form(10.0, description="Repetition penalty"),
    top_k: int = Form(50, description="Top-k sampling"),
    top_p: float = Form(0.85, description="Top-p sampling"),
    gpt_cond_len: int = Form(12, description="GPT conditioning length (seconds of ref audio to use)"),
):
    """
    RAW XTTS synthesis — bypasses FlowEdit entirely.
    
    Calls the Coqui XTTS model.inference() directly with no hooks,
    no HopfieldRefiner, no embedding manipulation. Use this to verify
    that the fine-tuned model itself produces correct pronunciation.
    """
    global pipeline
    if not pipeline or not pipeline.backbone:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
        shutil.copyfileobj(speaker_wav.file, temp_speaker)
        temp_speaker_path = temp_speaker.name

    output_path = tempfile.mktemp(suffix=".wav")

    try:
        import librosa
        import soundfile as sf
        import numpy as np

        # Preprocess speaker audio to 22050 Hz (XTTS input rate)
        processed_fd, processed_path = tempfile.mkstemp(suffix=".wav")
        os.close(processed_fd)
        y, sr = librosa.load(temp_speaker_path, sr=22050, mono=True)
        sf.write(processed_path, y, 22050, subtype='PCM_16')

        # Get speaker conditioning directly from the Coqui model
        model = pipeline.backbone.model
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[processed_path],
            gpt_cond_len=gpt_cond_len,
            gpt_cond_chunk_len=4,
            max_ref_length=30,
        )

        print(f"[RAW] text={text!r}, lang={language}, temp={temperature}, "
              f"rep_pen={repetition_penalty}, top_k={top_k}, top_p={top_p}, "
              f"gpt_cond_len={gpt_cond_len}")
        print(f"[RAW] gpt_cond_latent shape: {gpt_cond_latent.shape}")
        print(f"[RAW] speaker_embedding shape: {speaker_embedding.shape}")

        # Direct XTTS inference — NO FlowEdit hooks
        out = model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            length_penalty=1.0,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting=False,
        )

        wav = out["wav"]
        if isinstance(wav, torch.Tensor):
            wav_np = wav.squeeze().cpu().numpy()
        else:
            wav_np = np.array(wav)

        sf.write(output_path, wav_np, 24000)
        os.remove(processed_path)

        print(f"[RAW] Generated {len(wav_np)/24000:.2f}s of audio")

        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename="synthesized_raw.wav",
            background=None,
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

