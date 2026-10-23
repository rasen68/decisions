import asyncio
from collections import UserList
import unittest
from unittest.mock import AsyncMock, patch

from decisions import AsyncClient, Client, Image
from decisions.providers import OpenAI
import test_provider_inputs as provider_tests
from test_response_structure import PROVIDERS
from test_state_requests import METHODS, arguments, response_for


class ImageRequestTests(unittest.TestCase):
    def reject(self, provider, images, expected):
        helper = provider_tests.ProviderInputTests()
        for method in METHODS:
            argument, kwargs = arguments(method)
            helper.assert_no_send(provider, method, argument, {**kwargs, "images": images}, expected)

    def call(self, provider, images, method, asynchronous, state):
        argument, kwargs = arguments(method)
        response = response_for(provider, method)
        if asynchronous:
            async def run():
                async with AsyncClient(provider) as client:
                    with patch.object(client, "_send", AsyncMock(return_value=response)) as send:
                        await getattr(client, method)(argument, state, images=images, **kwargs)
                        send.assert_awaited_once()
                        return send.call_args.args[0].body
            return asyncio.run(run())
        with Client(provider) as client:
            with patch.object(client, "_send", return_value=response) as send:
                getattr(client, method)(argument, state, images=images, **kwargs)
                send.assert_called_once()
                return send.call_args.args[0].body

    def test_typesafe_adapter_rejects_images(self):
        self.reject(PROVIDERS[1], [Image(b"a", "image/png")], ValueError)

    def test_jeff_rejects_gif(self):
        self.reject(PROVIDERS[2], [Image(b"a", "image/gif")], ValueError)

    def test_images_require_a_sequence(self):
        for provider in PROVIDERS:
            for images in (None, "", b"", bytearray(), {}, set(), iter([]), Image(b"a", "image/png")):
                with self.subTest(provider=provider.name, images=images):
                    self.reject(provider, images, TypeError)

    def test_sequence_entries_require_image_objects(self):
        for provider in PROVIDERS:
            for value in (None, "image", b"image", {}, 1, True):
                with self.subTest(provider=provider.name, value=value):
                    self.reject(provider, [Image(b"a", "image/png"), value], TypeError)

    def test_mutated_images_are_revalidated(self):
        for provider in (PROVIDERS[0], PROVIDERS[2]):
            for field, value, expected in (("data", b"", ValueError), ("data", "image", TypeError), ("mime_type", None, TypeError), ("mime_type", "text/plain", ValueError)):
                image = Image(b"private-image", "image/png")
                setattr(image, field, value)
                with self.subTest(provider=provider.name, field=field, value=value):
                    self.reject(provider, [image], expected)
        image = Image(b"image", "image/png")
        images = [image]
        self.call(PROVIDERS[0], images, "predicate", False, None)
        images.append("not an Image")
        self.reject(PROVIDERS[0], images, TypeError)

    def test_supported_images_preserve_order_bytes_and_state(self):
        for provider in (PROVIDERS[0], PROVIDERS[2]):
            images = [Image(b"a", "image/png"), Image(b"b", "image/jpeg"), Image(b"c", "image/webp")]
            if isinstance(provider, OpenAI):
                images.append(Image(b"d", "image/gif"))
            expected = ["data:image/png;base64,YQ==", "data:image/jpeg;base64,Yg==", "data:image/webp;base64,Yw=="]
            if isinstance(provider, OpenAI):
                expected.append("data:image/gif;base64,ZA==")
            for method in METHODS:
                for asynchronous in (False, True):
                    state = " Text " if isinstance(provider, OpenAI) else {"nested": [True, None]}
                    with self.subTest(provider=provider.name, method=method, asynchronous=asynchronous):
                        body = self.call(provider, images, method, asynchronous, state)
                        if isinstance(provider, OpenAI):
                            parts = body["input"][0]["content"]
                            self.assertEqual(parts[0], {"type": "input_text", "text": state})
                            self.assertEqual([part["image_url"] for part in parts[1:]], expected)
                        else:
                            self.assertEqual(body["images"], expected)
                            self.assertIs(body["state"], state)
            for wrap in (tuple, UserList):
                body = self.call(provider, wrap(images), "predicate", False, None)
                if isinstance(provider, OpenAI):
                    self.assertEqual([part["image_url"] for part in body["input"][0]["content"]], expected)
                else:
                    self.assertEqual(body["images"], expected)
            self.assertEqual([image.data for image in images[:3]], [b"a", b"b", b"c"])

    def test_empty_sequences_remain_valid_for_all_providers(self):
        for provider in PROVIDERS:
            for images in ([], ()):
                for asynchronous in (False, True):
                    body = self.call(provider, images, "predicate", asynchronous, "Text")
                    if isinstance(provider, OpenAI):
                        self.assertEqual(body["input"], "Text")
                    else:
                        self.assertNotIn("images", body)


if __name__ == "__main__":
    unittest.main()
