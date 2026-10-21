"""Check consumer examples against a separately installed decisions package."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="Python containing the installed wheel")
    args = parser.parse_args()
    python = str(args.python.absolute())
    root = Path(__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"PYTHONPATH", "MYPYPATH"}}
    with tempfile.TemporaryDirectory(prefix="decisions-typing-") as directory:
        config = Path(directory) / "mypy.ini"
        config.write_text("[mypy]\n")
        installed = subprocess.run(
            [python, "-I", "-c",
             "from pathlib import Path; import decisions; "
             "print(Path(decisions.__file__).resolve())"],
            cwd=directory, env=environment, check=True, capture_output=True, text=True,
        )
        if Path(installed.stdout.strip()).is_relative_to(root / "decisions"):
            raise SystemExit("Install the wheel in a separate environment before checking typing")
        for fixture in (root / "tests" / "typing").glob("*.py"):
            shutil.copy(fixture, directory)
        subprocess.run(
            [sys.executable, "-m", "mypy", "--strict", "--no-incremental",
             "--python-executable", python, "--python-version", "3.11",
             "--config-file", str(config), "consumer.py", "invalid.py"],
            cwd=directory, env=environment, check=True,
        )


if __name__ == "__main__":
    main()
