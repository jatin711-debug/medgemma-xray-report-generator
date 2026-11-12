"""Bound request bodies and concurrent model work per server process."""
import asyncio

from fastapi import HTTPException
from starlette.responses import JSONResponse

MAX_REQUEST_BYTES = 11 * 1024 * 1024
_model_slot = asyncio.Semaphore(1)


class RequestSizeLimitMiddleware:
    """Reject oversized HTTP requests, including chunked bodies."""

    def __init__(self, app, max_bytes: int = MAX_REQUEST_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        length_headers = [value for name, value in scope['headers'] if name.lower() == b'content-length']
        try:
            if len(length_headers) > 1:
                raise ValueError('multiple content lengths')
            length = int(length_headers[0]) if length_headers else 0
            if length < 0:
                raise ValueError('negative content length')
        except ValueError:
            return await JSONResponse({'detail': 'Invalid Content-Length'}, status_code=400)(scope, receive, send)
        if length > self.max_bytes:
            return await JSONResponse({'detail': 'Request body too large'}, status_code=413)(scope, receive, send)

        received = 0
        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > self.max_bytes:
                    raise HTTPException(status_code=413, detail='Request body too large')
            return message
        return await self.app(scope, bounded_receive, send)


async def inference_slot():
    """Reject excess model work instead of queuing requests without a bound."""
    if _model_slot.locked():
        raise HTTPException(status_code=429, detail='Model is busy', headers={'Retry-After': '5'})
    async with _model_slot:
        yield
