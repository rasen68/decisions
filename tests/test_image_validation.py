from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decisions import Image


class ImageValidationTests(unittest.TestCase):
    def test_image_data_requires_bytes(self):
        for data in (None, "image", bytearray(b"image"), memoryview(b"image"), 1, True, [], {}):
            for construct in (lambda: Image(data, "image/png"), lambda: Image.from_bytes(data, mime_type="image/png")):
                with self.subTest(data=data):
                    with self.assertRaises(TypeError):
                        construct()

    def test_image_data_must_not_be_empty(self):
        for construct in (lambda: Image(b"", "image/png"), lambda: Image.from_bytes(b"", mime_type="image/png")):
            with self.assertRaises(ValueError):
                construct()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.png"
            path.write_bytes(b"")
            with self.assertRaises(ValueError):
                Image.from_path(path)

    def test_mime_type_requires_a_string(self):
        for mime_type in (None, 1, True, b"image/png", [], {}):
            with self.subTest(mime_type=mime_type):
                with self.assertRaises(TypeError):
                    Image(b"image", mime_type)
                with self.assertRaises(TypeError):
                    Image.from_bytes(b"image", mime_type=mime_type)

    def test_mime_type_must_be_supported_without_coercion(self):
        for mime_type in ("", " ", "image/unknown", "text/plain", "image/svg+xml", "image/bmp", "image/jpg", "IMAGE/PNG", " image/png", "image/png; charset=utf-8"):
            with self.subTest(mime_type=mime_type):
                with self.assertRaises(ValueError):
                    Image(b"image", mime_type)

    def test_supported_metadata_preserves_bytes_without_decoding(self):
        data = b"deliberately not a real image"
        for mime_type in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            image = Image.from_bytes(data, mime_type=mime_type)
            self.assertIs(image.data, data)
            self.assertIs(image.mime_type, mime_type)

    def test_unknown_filename_inference_fails_and_override_works(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photo.no-known-image-extension"
            path.write_bytes(b"image")
            with patch("decisions.types.inputs.mimetypes.guess_type", return_value=(None, None)):
                with self.assertRaises(ValueError):
                    Image.from_path(path)
            self.assertEqual(Image.from_path(path, mime_type="image/png"), Image(b"image", "image/png"))

    def test_filename_inference_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix, mime_type in (("png", "image/png"), ("jpg", "image/jpeg"), ("jpeg", "image/jpeg"), ("webp", "image/webp"), ("gif", "image/gif")):
                path = Path(directory) / f"photo.{suffix}"
                path.write_bytes(b"image")
                self.assertEqual(Image.from_path(path).mime_type, mime_type)
                self.assertEqual(Image.from_path(str(path), mime_type="image/webp").mime_type, "image/webp")
            path = Path(directory) / "photo.txt"
            path.write_bytes(b"image")
            with self.assertRaises(ValueError):
                Image.from_path(path)
            self.assertEqual(Image.from_path(path, mime_type="image/png").mime_type, "image/png")

    def test_file_errors_are_not_wrapped(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                Image.from_path(Path(directory) / "missing.png")
            with self.assertRaises(IsADirectoryError):
                Image.from_path(directory)
        original = PermissionError("denied")
        with patch.object(Path, "read_bytes", side_effect=original):
            with self.assertRaises(PermissionError) as caught:
                Image.from_path("photo.png")
        self.assertIs(caught.exception, original)


if __name__ == "__main__":
    unittest.main()
