"""Entry point to download the CaBuAr dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pyrosegnet.data.download import download_cabuar

if __name__ == "__main__":
    download_cabuar()
