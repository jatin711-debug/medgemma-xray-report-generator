import asyncio
import io
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile
from PIL import Image

from src.services.image_input import PublicResolver, decode_image, fetch_image, read_upload, validate_public_url


def png_bytes():
    output = io.BytesIO()
    Image.new('RGB', (4, 4)).save(output, format='PNG')
    return output.getvalue()


class ImageValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_size_and_format_validation(self):
        upload = UploadFile(filename='x.png', file=io.BytesIO(png_bytes()))
        image, metadata = await read_upload(upload, 1024)
        self.assertEqual(image.size, (4, 4))
        self.assertEqual(metadata['format'], 'PNG')
        with self.assertRaises(HTTPException) as result:
            await read_upload(UploadFile(filename='x.png', file=io.BytesIO(png_bytes())), 5)
        self.assertEqual(result.exception.status_code, 413)
        with self.assertRaises(HTTPException) as result:
            decode_image(b'not an image')
        self.assertEqual(result.exception.status_code, 400)

    async def test_image_pixel_limit(self):
        with patch('src.services.image_input.MAX_IMAGE_PIXELS', 10):
            with self.assertRaises(HTTPException) as result:
                decode_image(png_bytes())
        self.assertEqual(result.exception.status_code, 413)

    async def test_private_url_and_credentials_rejected(self):
        for url in ('http://127.0.0.1/x', 'http://[::1]/x', 'http://localhost/x',
                    'http://[64:ff9b::a00:1]/x', 'http://[2002:a00:1::]/x',
                    'http://[64:ff9b:1::a00:1]/x',
                    'http://user:pass@example.com/x', 'file:///etc/passwd'):
            with self.subTest(url=url), self.assertRaises(HTTPException):
                validate_public_url(url)
        validate_public_url('https://example.com/test.png')
        validate_public_url('https://[2606:4700:4700::1111]/test.png')

    async def test_dns_resolver_rejects_mixed_public_and_private_addresses(self):
        resolver = PublicResolver()
        addresses = [{'host': '8.8.8.8'}, {'host': '127.0.0.1'}]
        try:
            with patch.object(resolver._resolver, 'resolve', AsyncMock(return_value=addresses)):
                with self.assertRaises(OSError):
                    await resolver.resolve('example.com')
        finally:
            await resolver.close()

    async def test_dns_resolver_rejects_ipv6_translation_addresses(self):
        resolver = PublicResolver()
        try:
            with patch.object(resolver._resolver, 'resolve', AsyncMock(return_value=[{'host': '64:ff9b::a00:1'}])):
                with self.assertRaises(OSError):
                    await resolver.resolve('example.com')
        finally:
            await resolver.close()

    async def test_redirect_to_private_ip_rejected_before_request(self):
        calls = []
        class Response:
            status = 302
            headers = {'Location': 'http://127.0.0.1/private'}
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
        class Session:
            def __init__(self, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            def get(self, url, **kwargs):
                calls.append(url)
                return Response()
        with patch('src.services.image_input.aiohttp.ClientSession', Session):
            with self.assertRaises(HTTPException) as result:
                await fetch_image('https://example.com/image', 1024)
        self.assertEqual(result.exception.status_code, 400)
        self.assertEqual(calls, ['https://example.com/image'])

    async def test_streamed_body_respects_limit_without_content_length(self):
        class Content:
            async def iter_chunked(self, size):
                yield b'a' * 5
                yield b'b' * 6
        class Response:
            status = 200
            headers = {'Content-Type': 'image/png'}
            content_length = None
            content = Content()
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
        class Session:
            def __init__(self, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            def get(self, *args, **kwargs): return Response()
        with patch('src.services.image_input.aiohttp.ClientSession', Session):
            with self.assertRaises(HTTPException) as result:
                await fetch_image('https://example.com/image', 10)
        self.assertEqual(result.exception.status_code, 413)
