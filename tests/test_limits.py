import unittest

import httpx
from fastapi import Depends, FastAPI, Request, File, UploadFile

from src.api.limits import RequestSizeLimitMiddleware, inference_slot


class RequestLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_content_length_and_chunked_body_are_limited(self):
        app = FastAPI()
        app.add_middleware(RequestSizeLimitMiddleware, max_bytes=8)

        @app.post('/')
        async def consume(request: Request):
            return {'size': len(await request.body())}

        @app.post('/upload')
        async def upload(file: UploadFile = File(...)):
            return {'filename': file.filename}

        async def stream():
            yield b'12345'
            yield b'67890'

        async def multipart_stream():
            yield b'--test\r\n'
            yield b'Content-Disposition: form-data; name="file"; filename="x"\r\n'
            yield b'\r\n123456789\r\n--test--\r\n'

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            self.assertEqual((await client.post('/', content=b'123456789')).status_code, 413)
            self.assertEqual((await client.post('/', content=stream())).status_code, 413)
            self.assertEqual((await client.post('/upload', content=multipart_stream(), headers={
                'content-type': 'multipart/form-data; boundary=test'})).status_code, 413)
            self.assertEqual((await client.post('/', content=b'1234')).status_code, 200)

    async def test_inference_capacity_rejects_additional_work(self):
        first = inference_slot()
        second = inference_slot()
        await anext(first)
        try:
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as result:
                await anext(second)
            self.assertEqual(result.exception.status_code, 429)
        finally:
            await first.aclose()
