import torch
import logging
import os
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.WARNING)

from flowedit.backbone.f5tts_wrapper import F5TTSBackbone

def test_gradient():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    backbone = F5TTSBackbone(device=device)
    backbone.load()
    
    dummy_wav_path = "dummy_ref.wav"
    sf.write(dummy_wav_path, np.zeros(24000), 24000)
    
    speaker_cond = backbone.get_speaker_embedding(dummy_wav_path, "en", "Mrunmayee Sakharwade")
    
    base = torch.randn(1, 25, 512, device=device)
    delta = torch.zeros_like(base, requires_grad=True)
    perturbed = base + delta
    
    print("\n=== Test A & B: Gradient Flow ===")
    result = backbone.compute_optimization_loss(
        perturbed_embeddings=perturbed,
        ref_audio_path=dummy_wav_path,
        speaker_conditioning=speaker_cond,
        text="Mrunmayee Sakharwade",
        language="en"
    )
    
    loss = result["loss"]
    print(f"LOSS: {loss.item()}")
    print(f"loss.requires_grad: {loss.requires_grad}")
    print(f"loss.grad_fn: {loss.grad_fn}")
    
    loss.backward()
    
    if delta.grad is not None:
        print(f"delta.grad is not None: True")
        print(f"delta.grad.norm(): {delta.grad.norm().item()}")
    else:
        print("delta.grad is None! The optimization will fail.")
        
    print("\n=== Test C: Perturbation Impact ===")
    with torch.no_grad():
        base_out = backbone.synthesize_from_embeddings(
            text_embeddings=base,
            speaker_conditioning=speaker_cond,
            text="Mrunmayee Sakharwade",
            language="en"
        )
        base_waveform = base_out[0] if isinstance(base_out, tuple) else base_out
        
        test_delta = torch.randn_like(base) * 0.1
        pert_out = backbone.synthesize_from_embeddings(
            text_embeddings=base + test_delta,
            speaker_conditioning=speaker_cond,
            text="Mrunmayee Sakharwade",
            language="en"
        )
        pert_waveform = pert_out[0] if isinstance(pert_out, tuple) else pert_out
        
        diff = torch.norm(base_waveform - pert_waveform).item()
        print(f"|| waveform(base) - waveform(base + delta) ||_2 = {diff}")
        if diff > 0:
            print("Perturbation successfully changed the audio output!")
        else:
            print("WARNING: Audio output did NOT change. The injection path is broken.")

    print("\n=== Test D: Finite Difference Check ===")
    epsilon = 1e-3
    # Pick a random element where the gradient is significant
    if delta.grad is not None and delta.grad.norm().item() > 0:
        # Find index with max gradient magnitude to avoid precision issues
        max_idx = torch.argmax(torch.abs(delta.grad))
        i, j, k = np.unravel_index(max_idx.item(), delta.grad.shape)
        
        grad_autograd = delta.grad[i, j, k].item()
        
        # L(delta + eps)
        with torch.no_grad():
            delta_plus = delta.detach().clone()
            delta_plus[i, j, k] += epsilon
            res_plus = backbone.compute_optimization_loss(
                perturbed_embeddings=base + delta_plus,
                ref_audio_path=dummy_wav_path,
                speaker_conditioning=speaker_cond,
                text="Mrunmayee Sakharwade",
                language="en"
            )
            l_plus = res_plus["loss"].item()
            
            # L(delta - eps)
            delta_minus = delta.detach().clone()
            delta_minus[i, j, k] -= epsilon
            res_minus = backbone.compute_optimization_loss(
                perturbed_embeddings=base + delta_minus,
                ref_audio_path=dummy_wav_path,
                speaker_conditioning=speaker_cond,
                text="Mrunmayee Sakharwade",
                language="en"
            )
            l_minus = res_minus["loss"].item()
            
        grad_fd = (l_plus - l_minus) / (2 * epsilon)
        
        print(f"Testing gradient at index ({i}, {j}, {k})")
        print(f"Autograd gradient: {grad_autograd:.6f}")
        print(f"Finite difference: {grad_fd:.6f}")
        
        if abs(grad_fd) > 1e-9:
            rel_error = abs(grad_autograd - grad_fd) / (abs(grad_fd) + 1e-9) * 100
            print(f"Relative error: {rel_error:.2f}%")
        else:
            print("Gradient is too close to zero to compute relative error accurately.")
            print(f"Absolute difference: {abs(grad_autograd - grad_fd):.6e}")
    else:
        print("Cannot perform finite difference check without valid gradients.")

    if os.path.exists(dummy_wav_path):
        os.remove(dummy_wav_path)

if __name__ == "__main__":
    test_gradient()
