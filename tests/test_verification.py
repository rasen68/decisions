from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OfflineRunnerTests(unittest.TestCase):
    def run_fixture(self, source):
        with tempfile.TemporaryDirectory() as directory:
            tests = Path(directory) / "tests"
            tests.mkdir()
            (tests / "test_fixture.py").write_text(textwrap.dedent(source))
            return subprocess.run(
                [sys.executable, "-I", "-B", str(ROOT / "scripts/test_offline.py"),
                 "--tests", str(tests)],
                cwd=directory, capture_output=True, text=True, timeout=30,
            )

    def test_blocks_connect_and_connect_ex(self):
        result = self.run_fixture('''
            import socket
            import unittest

            class SocketTests(unittest.TestCase):
                def test_connections_are_blocked(self):
                    with socket.socket() as connection:
                        for connect in (connection.connect, connection.connect_ex):
                            with self.assertRaisesRegex(AssertionError, "network access"):
                                connect(("127.0.0.1", 9))
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Ran 1 test", result.stderr)

    def test_propagates_suite_failure(self):
        result = self.run_fixture('''
            import unittest

            class FailureTests(unittest.TestCase):
                def test_failure(self):
                    self.fail("deliberate fixture failure")
        ''')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("deliberate fixture failure", result.stderr)

    def test_rejects_empty_discovery(self):
        result = self.run_fixture("# No tests\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No tests discovered", result.stderr)

    def test_installed_mode_requires_isolation(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/test_offline.py"), "--installed"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--installed requires Python's -I flag", result.stderr)

    def test_installed_mode_rejects_source_import(self):
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", "\n".join([
                "import runpy, sys",
                f"sys.path.insert(0, {str(ROOT)!r})",
                "sys.argv = ['test_offline.py', '--installed']",
                f"runpy.run_path({str(ROOT / 'scripts/test_offline.py')!r}, run_name='__main__')",
            ])], capture_output=True, text=True, timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("imported the source checkout", result.stderr)
