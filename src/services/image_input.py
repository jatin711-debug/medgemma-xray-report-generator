"""Validation and bounded retrieval of research image inputs."""

import asyncio
import io
import ipaddress
import socket
import warnings
from urllib.parse import urljoin, urlsplit

import aiohttp
from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "TIFF", "BMP", "WEBP"}
MAX_REDIRECTS = 3
IPV6_TRANSLATION_PREFIXES = (
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
    ipaddress.ip_network("2002::/16"),
    ipaddress.ip_network("2001::/32"),
)


def is_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Allow global unicast addresses, excluding IPv6 transition mechanisms.

    Translation and tunneling prefixes can route to an embedded private IPv4
    destination even when Python considers the IPv6 address global.
    """
    if not address.is_global or address.is_multicast:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return is_public_address(address.ipv4_mapped)
        if any(address in prefix for prefix in IPV6_TRANSLATION_PREFIXES):
            return False
    return True


def decode_image(data: bytes) -> tuple[Image.Image, str]:
    """Decode a supported image, enforcing a pixel limit before conversion."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                image_format = source.format
                if image_format not in ALLOWED_FORMATS:
                    raise HTTPException(status_code=400, detail="Unsupported image format")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=413, detail="Image dimensions are too large")
                image = source.convert("RGB")
                image.load()
                return image, image_format
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=413, detail="Image dimensions are too large") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image format") from exc


def validate_public_url(url: str) -> None:
    """Reject non-HTTP URLs, embedded credentials, and non-public IP literals."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        _ = parsed.port  # Validate port syntax and range.
        if parsed.scheme.lower() not in {"http", "https"} or not host or parsed.username is not None or parsed.password is not None:
            raise ValueError("Invalid URL")
        if "%" in host or host.lower().rstrip(".").endswith(".localhost") or host.lower().rstrip(".") == "localhost":
            raise ValueError("Non-public host")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return  # DNS is checked by PublicResolver at connection time.
        if not is_public_address(address):
            raise ValueError("Non-public address")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="URL must use a public HTTP(S) address without credentials") from exc


class PublicResolver(aiohttp.abc.AbstractResolver):
    """Resolve hostnames only to globally routable addresses, including redirects."""

    def __init__(self):
        self._resolver = aiohttp.resolver.DefaultResolver()

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_UNSPEC):
        addresses = await self._resolver.resolve(host, port, family)
        if not addresses or any(not is_public_address(ipaddress.ip_address(item["host"])) for item in addresses):
            raise OSError("Host resolves to a non-public address")
        return addresses

    async def close(self):
        await self._resolver.close()


async def read_upload(file: UploadFile, max_bytes: int) -> tuple[Image.Image, dict]:
    """Read no more than the byte limit and validate the uploaded image."""
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the upload size limit")
    image, image_format = await run_in_threadpool(decode_image, data)
    return image, {"source": "upload", "filename": file.filename, "size": len(data),
                   "format": image_format, "mode": image.mode, "dimensions": image.size,
                   "content_type": file.content_type}


async def fetch_image(url: str, max_bytes: int) -> tuple[Image.Image, dict]:
    """Download an image with address checks, bounded redirects, and streaming limits."""
    timeout = aiohttp.ClientTimeout(total=30)
    resolver = PublicResolver()
    connector = aiohttp.TCPConnector(resolver=resolver, use_dns_cache=False)
    try:
        async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:
            current_url = url
            for redirect_count in range(MAX_REDIRECTS + 1):
                validate_public_url(current_url)
                async with session.get(current_url, headers={"Accept": "image/*"}, allow_redirects=False) as response:
                    if response.status in {301, 302, 303, 307, 308}:
                        location = response.headers.get("Location")
                        if not location or redirect_count == MAX_REDIRECTS:
                            raise HTTPException(status_code=400, detail="Invalid or excessive image redirects")
                        current_url = urljoin(current_url, location)
                        continue
                    if response.status != 200:
                        raise HTTPException(status_code=400, detail="Image URL returned an unsuccessful response")
                    content_type = response.headers.get("Content-Type", "").lower()
                    if not content_type.startswith("image/"):
                        raise HTTPException(status_code=400, detail="URL did not return an image")
                    if response.content_length is not None and response.content_length > max_bytes:
                        raise HTTPException(status_code=413, detail="Image exceeds the download size limit")
                    data = bytearray()
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        data.extend(chunk)
                        if len(data) > max_bytes:
                            raise HTTPException(status_code=413, detail="Image exceeds the download size limit")
                    image, image_format = await run_in_threadpool(decode_image, bytes(data))
                    return image, {"source": "url", "size": len(data), "format": image_format,
                                   "mode": image.mode, "dimensions": image.size, "content_type": content_type}
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Could not fetch image from a public address") from exc
    finally:
        await resolver.close()
