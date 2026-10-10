"""
Speaker Reference Voice Locator and Preset Registry for FlowEdit.

Resolves preset deployment reference voices (e.g. Blessing, Michael)
across project candidate directories, environment variables, and resources.
"""

import os
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

MICHAEL_VOICE_FILE_PATH: Optional[str] = os.environ.get("FLOWEDIT_MICHAEL_VOICE", "/home/rsurya/projects/flow_edit/michael.wav")
BLESSING_VOICE_FILE_PATH: Optional[str] = os.environ.get("FLOWEDIT_BLESSING_VOICE", "/home/rsurya/projects/flow_edit/blessing.wav")

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
    if is_male and MICHAEL_VOICE_FILE_PATH and os.path.isfile(MICHAEL_VOICE_FILE_PATH) and os.path.getsize(MICHAEL_VOICE_FILE_PATH) > 1000:
        return os.path.abspath(MICHAEL_VOICE_FILE_PATH)
    if not is_male and BLESSING_VOICE_FILE_PATH and os.path.isfile(BLESSING_VOICE_FILE_PATH) and os.path.getsize(BLESSING_VOICE_FILE_PATH) > 1000:
        return os.path.abspath(BLESSING_VOICE_FILE_PATH)

    voice_key = HARDCODED_DEFAULT_VOICES.get(target_lower, "michael" if is_male else "blessing")
    filename = f"{voice_key}.wav"

    pkg_utils_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_flowedit_dir = os.path.dirname(pkg_utils_dir)
    flowedit_root = os.path.abspath(os.path.join(pkg_flowedit_dir, ".."))
    workspace_root = os.path.abspath(os.path.join(flowedit_root, ".."))

    candidate_locations = [
        # Primary confirmed IIT server paths
        f"/home/rsurya/projects/flow_edit/{filename}",
        f"/home/rsurya/projects/flow_edit/Flowedit/model/{filename}",
        f"/home/rsurya/projects/flow_edit/Flowedit/deploy_voices/{filename}",
        f"/home/rsurya/projects/flow_edit/Flowedit/flowedit/resources/{filename}",
        os.path.join(flowedit_root, "deploy_voices", filename),
        os.path.abspath(os.path.join(pkg_flowedit_dir, "resources", filename)),
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
        res_default = os.path.abspath(os.path.join(pkg_flowedit_dir, "resources", "default_speaker.wav"))
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
    pkg_utils_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_flowedit_dir = os.path.dirname(pkg_utils_dir)
    flowedit_root = os.path.abspath(os.path.join(pkg_flowedit_dir, ".."))
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
