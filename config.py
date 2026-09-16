import os
import platform
from pathlib import Path

# Repository root (which is the directory containing this config.py)
ROOT_DIR = Path(__file__).resolve().parent

# Data/Results paths - can be overridden by environment variables
DATA_DIR = Path(os.getenv("JASEE_DATA_DIR", ROOT_DIR / "data"))
RESULTS_DIR = Path(os.getenv("JASEE_RESULTS_DIR", ROOT_DIR / "results"))

def korean_font():
    """Returns the path to a Korean font depending on the OS, or None if not found."""
    system = platform.system()
    
    if system == "Windows":
        font_path = "C:\\Windows\\Fonts\\malgun.ttf"
        if os.path.exists(font_path):
            return font_path
    elif system == "Darwin": # macOS
        font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
        if os.path.exists(font_path):
            return font_path
    elif system == "Linux":
        # Check a few common Linux font paths
        for p in [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]:
            if os.path.exists(p):
                return p
    return None
