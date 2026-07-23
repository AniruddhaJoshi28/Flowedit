from setuptools import setup, find_packages

setup(
    name="flowedit",
    version="0.1.0",
    description="FlowEdit: Lifelong Pronunciation Adaptation for TTS via Associative Memory",
    author="Info Origin Technologies",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "TTS>=0.22.0",
        "openai-whisper>=20231117",
        "stable-ts>=2.15.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
)
