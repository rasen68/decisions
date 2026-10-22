"""Build an sdist and its wheel, then test a fresh installation and consumer typing."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--httpx", default="httpx>=0.27,<1",
                        help="HTTPX requirement to install alongside the wheel")
    args = parser.parse_args()
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"PYTHONPATH", "MYPYPATH"}}
    environment.update(OPENAI_API_KEY="", TYPESAFE_API_KEY="", PYTHONDONTWRITEBYTECODE="1")

    with tempfile.TemporaryDirectory(prefix="decisions-install-") as directory:
        work = Path(directory)
        dist = work / "dist"
        # With no format flags, build makes the wheel from the generated sdist.
        subprocess.run([sys.executable, "-m", "build", "--outdir", str(dist), str(ROOT)],
                       cwd=work, env=environment, check=True)
        wheels = list(dist.glob("*.whl"))
        if len(wheels) != 1 or len(list(dist.glob("*.tar.gz"))) != 1:
            raise RuntimeError("Expected one sdist and one wheel")

        install_env = work / "env"
        venv.EnvBuilder(with_pip=True).create(install_env)
        bin_dir = install_env / ("Scripts" if os.name == "nt" else "bin")
        python = str(bin_dir / ("python.exe" if os.name == "nt" else "python"))
        subprocess.run([python, "-m", "pip", "--disable-pip-version-check", "install",
                        str(wheels[0]), args.httpx], cwd=work, env=environment, check=True)
        subprocess.run([python, "-I", "-c", "\n".join([
            "from importlib.resources import files",
            "from pathlib import Path",
            "import sys, decisions, httpx",
            "assert Path(decisions.__file__).resolve().is_relative_to(Path(sys.prefix))",
            "assert files('decisions').joinpath('py.typed').is_file()",
            "print(f'Installed decisions {decisions.__version__}, HTTPX {httpx.__version__}')",
        ])], cwd=work, env=environment, check=True)
        subprocess.run([python, "-I", "-B", str(ROOT / "scripts/test_offline.py"), "--installed"],
                       cwd=work, env=environment, check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/check_typing.py"), "--python", python],
                       cwd=work, env=environment, check=True)

        command = str(bin_dir / ("decisions.exe" if os.name == "nt" else "decisions"))
        for argument in ("--version", "--help"):
            result = subprocess.run([command, argument], cwd=work, env=environment,
                                    capture_output=True, text=True, check=True)
            if result.stderr:
                raise RuntimeError(result.stderr)
        result = subprocess.run([command, "-p", "Ready?", "-t", "Text"],
                                cwd=work, env=environment, capture_output=True, text=True)
        if result.returncode != 2 or result.stdout or "OPENAI_API_KEY" not in result.stderr \
                or "Traceback" in result.stderr:
            raise RuntimeError("Installed command did not handle missing authentication correctly")
        print("Installed package, consumer typing, and command checks passed")


if __name__ == "__main__":
    main()
