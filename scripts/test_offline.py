"""Run tests with socket connections blocked, against source or an installed wheel."""

import argparse
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def blocked(*args, **kwargs):
    raise AssertionError("Unexpected network access in offline tests")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true",
                        help="Test the installed package; requires Python's -I flag")
    parser.add_argument("--tests", type=Path, default=ROOT / "tests")
    parser.add_argument("--pattern", default="test_*.py")
    args = parser.parse_args()
    if args.installed:
        if not sys.flags.isolated:
            parser.error("--installed requires Python's -I flag")
        import decisions
        if Path(decisions.__file__).resolve().is_relative_to(ROOT / "decisions"):
            parser.error("Expected an installed wheel, but imported the source checkout")
    else:
        sys.path.insert(0, str(ROOT))

    with patch.object(socket.socket, "connect", blocked), \
            patch.object(socket.socket, "connect_ex", blocked):
        suite = unittest.defaultTestLoader.discover(str(args.tests.absolute()), pattern=args.pattern)
        if not suite.countTestCases():
            parser.error("No tests discovered")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
