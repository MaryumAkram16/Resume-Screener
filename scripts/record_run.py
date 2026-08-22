"""Record a portable model-run manifest without requiring MLflow or W&B.

Usage: python scripts/record_run.py --artifact-dir artifacts/stage2 --metrics artifacts/stage2/metrics.json
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--artifact-dir", type=Path, required=True); p.add_argument("--metrics", type=Path, required=True); p.add_argument("--output", type=Path); args = p.parse_args()
    try: revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError): revision = "unavailable"
    artifacts = {path.name: sha256(path) for path in sorted(args.artifact_dir.iterdir()) if path.is_file()}
    payload = {"recorded_at": datetime.now(timezone.utc).isoformat(), "git_revision": revision, "metrics_file": str(args.metrics), "artifacts": artifacts, "metrics": json.loads(args.metrics.read_text())}
    output = args.output or args.artifact_dir / "run_manifest.json"; output.write_text(json.dumps(payload, indent=2) + "\n"); print(json.dumps(payload, indent=2))

if __name__ == "__main__": main()
