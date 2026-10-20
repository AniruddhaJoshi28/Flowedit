import pytest

from flowedit.backbone.checkpoint_identity import CheckpointIdentityError, inspect_f5_safetensors


def test_auto_identity_rejects_opaque_pytorch_checkpoint():
    candidate = "untrusted_model.pth"
    with pytest.raises(CheckpointIdentityError, match="only .safetensors"):
        inspect_f5_safetensors(candidate)


def test_auto_identity_rejects_missing_checkpoint():
    with pytest.raises(CheckpointIdentityError):
        inspect_f5_safetensors("missing.safetensors")
