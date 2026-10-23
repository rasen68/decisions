"""Check internal component registrations and incompatible composition examples."""

from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--no-incremental",
         "--python-version", "3.11", "--follow-imports=silent",
         "decisions/providers/composition.py", "decisions/providers/integrations.py",
         "tests/composition_typing/contracts.py"],
        cwd=root, check=True,
    )


if __name__ == "__main__":
    main()
