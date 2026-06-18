import sys
from pathlib import Path

# Add src directory to PYTHONPATH
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from anomaly_detection.app import create_app

app = create_app()
