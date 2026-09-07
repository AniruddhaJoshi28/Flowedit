"""
Environment configuration loader for FlowEdit.
Loads .env files automatically, supporting python-dotenv with an internal fallback parser.
"""

import os
import logging

logger = logging.getLogger(__name__)


def load_flowedit_env() -> bool:
    """Load variables from .env file into os.environ if not already present."""
    # Try python-dotenv first
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Pure Python fallback parser
    search_dirs = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")),
    ]
    loaded = False
    seen = set()
    for d in search_dirs:
        if d in seen:
            continue
        seen.add(d)
        env_file = os.path.join(d, ".env")
        if os.path.isfile(env_file):
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
                loaded = True
                break
            except Exception as e:
                logger.debug(f"Could not parse .env file at {env_file}: {e}")

    return loaded
