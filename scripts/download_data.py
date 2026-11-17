"""Download UCI Adult (adult.data) into data/ and verify its SHA-256."""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
SHA256 = "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d"
DEST = Path("data/adult.data")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    DEST.parent.mkdir(exist_ok=True)
    if not DEST.exists():
        urllib.request.urlretrieve(URL, DEST)
    got = sha256(DEST)
    if got != SHA256:
        print(f"checksum mismatch: {got} != {SHA256}", file=sys.stderr)
        return 1
    print(f"ok: {DEST} ({got[:12]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
