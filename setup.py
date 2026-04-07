import importlib
from os import system
from sys import executable

from setuptools import find_packages, setup

# GroundingDINO requires torch to be present at *build* time (not just runtime).
# Only attempt the pre-install when torch isn't already importable, and swallow
# failures so the rest of setup.py can continue (e.g. inside pip's isolated
# build environment where pip itself may be absent).
try:
    importlib.import_module("torch")
except ImportError:
    system(f"{executable} -m pip install torch")

setup(
    name="octo-pearl",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "anthropic",
        "clip @ git+https://github.com/openai/CLIP.git",
        "clip-text-decoder",
        "flair",
        "groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git",
        "numpy==1.26.4",
        "opencv-python",
        "openai>=1.0.0",
        "Pillow",
        "python-dotenv",
        "ram @ git+https://github.com/xinyu1205/recognize-anything.git",
        "segment_anything @ git+https://github.com/facebookresearch/segment-anything.git",
        "tenacity",
    ],
)
