#!/usr/bin/env python3
import asyncio
import collections
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import aiohttp
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("vllm-router")

LISTEN_HOST = os.environ.get("ROUTER_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("ROUTER_PORT", "8080"))
MAX_ACTIVE_PER_NODE = int(os.environ.get("ROUTER_MAX_ACTIVE_PER_NODE", "12"))
SESSION_TTL_SECONDS = int(os.environ.get("ROUTER_SESSION_TTL", "7200"))
MAX_SESSIONS = int(os.environ.get("ROUTER_MAX_SESSIONS", "20000"))
HEALTH_CHECK_INTERVAL = float(os.environ.get("ROUTER_HEALTH_INTERVAL", "3.0"))

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}

SESSION_HEADERS = (
    "x-session-id",
    "x-conversation-id",
    "session-id",
    "conversation-id",
    "x-cache-key",
)


@dataclass
class Backend:
    name: str
    url: str
    load_bias: int = 0
    healthy: bool = True
    active_requests: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    last_error: str = ""
    last_check_ts: float = field(default_factory=time.time)


LJSAN1_LOAD_BIAS = int(os.environ.get("ROUTER_LJSAN1_LOAD_BIAS", "1"))

ALL_BACKENDS: List[Backend] = [
    Backend("ljsan2", "http://192.168.1.14:8000", load_bias=0),
    Backend("ljsan3", "http://192.168.1.15:8000", load_bias=0),
    Backend("ljsan4", "http://192.168.1.16:8000", load_bias=0),
    Backend("ljsan1", "http://192.168.1.13:8000", load_bias=LJSAN1_LOAD_BIAS),
]
_enabled_names = {
    x.strip()
    for x in os.environ.get("ROUTER_BACKENDS", "ljsan1,ljsan2,ljsan3,ljsan4").split(",")
    if x.strip()
}
BACKENDS: List[Backend] = [b for b in ALL_BACKENDS if b.name in _enabled_names] or ALL_BACKENDS
BACKEND_BY_NAME: Dict[str, Backend] = {b.name: b for b in BACKENDS}


class SessionStore:
    def __init__(self, max_size: int = MAX_SESSIONS, ttl: int = SESSION_TTL_SECONDS):
        self.max_size = max_size
        self.ttl = ttl
        self._store: "collections.OrderedDict[str, Tuple[str, float]]" = collections.OrderedDict()

    def get(self, key: str) -> Optional[str]:
        now = time.time()
        item = self._store.get(key)
        if item is None:
            return None
        backend_name, ts = item
        if now - ts > self.ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        self._store[key] = (backend_name, now)
        return backend_name

    def put(self, key: str, backend_name: str) -> None:
        now = time.time()
        self._store[key] = (backend_name, now)
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)


SESSION_STORE = SessionStore()
RR_COUNTER = 0


def _normalize_content(content) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(content)


def extract_routing_key(headers: web.BaseRequest.headers, body_bytes: bytes) -> Optional[str]:
    for h in SESSION_HEADERS:
        val = headers.get(h, "").strip()
        if val:
            return f"hdr:{ hashlib.sha256(val.encode('utf-8', 'ignore')).hexdigest()[:20] }"

    if not body_bytes:
        return None

    try:
        data = json.loads(body_bytes)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    for field_name in ("session_id", "conversation_id", "prompt_cache_key", "user"):
        val = data.get(field_name)
        if isinstance(val, str) and val.strip():
            return f"fld:{ hashlib.sha256(val.strip().encode('utf-8', 'ignore')).hexdigest()[:20] }"

    for nested in ("metadata", "extra_body"):
        sub = data.get(nested)
        if isinstance(sub, dict):
            for field_name in ("session_id", "conversation_id", "prompt_cache_key"):
                val = sub.get(field_name)
                if isinstance(val, str) and val.strip():
                    return f"meta:{ hashlib.sha256(val.strip().encode('utf-8', 'ignore')).hexdigest()[:20] }"

    messages = data.get("messages")
    if isinstance(messages, list) and messages:
        # Hash messages up to and including the first user message so all turns
        # of the same multi-turn conversation share the exact same key, while
        # different conversations sharing the same system prompt get distinct keys.
        prefix_parts = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", ""))
            content = _normalize_content(msg.get("content", ""))
            prefix_parts.append(f"{role}:{content}")
            if role == "user":
                break
        if prefix_parts:
            joined = "\n||\n".join(prefix_parts)
            return f"pfx:{ hashlib.sha256(joined.encode('utf-8', 'ignore')).hexdigest()[:20] }"

    prompt = data.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        head = prompt[:512]
        return f"pmt:{ hashlib.sha256(head.encode('utf-8', 'ignore')).hexdigest()[:20] }"

    return None


def hrw_score(key: str, backend_name: str) -> int:
    digest = hashlib.sha256(f"{key}@{backend_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def select_backend(routing_key: Optional[str], exclude: set) -> Tuple[Backend, str]:
    global RR_COUNTER
    candidates = [b for b in BACKENDS if b.healthy and b.name not in exclude]
    if not candidates:
        candidates = [b for b in BACKENDS if b.name not in exclude] or list(BACKENDS)

    if routing_key:
        cached_name = SESSION_STORE.get(routing_key)
        if cached_name and cached_name not in exclude:
            cached_backend = BACKEND_BY_NAME.get(cached_name)
            if cached_backend and cached_backend.healthy:
                if cached_backend.active_requests < MAX_ACTIVE_PER_NODE:
                    return cached_backend, "hit"
                min_active = min(b.active_requests for b in candidates)
                if cached_backend.active_requests <= min_active:
                    return cached_backend, "hit"

        # New session or previous backend overloaded/unhealthy:
        # Pick the least-loaded healthy backend, breaking ties via round-robin so both
        # sequential and concurrent new sessions spread evenly across all 4 nodes.
        n = len(BACKENDS)
        order = {BACKENDS[(RR_COUNTER + i) % n].name: i for i in range(n)}
        RR_COUNTER = (RR_COUNTER + 1) % n
        chosen = min(
            candidates,
            key=lambda b: (b.active_requests + b.load_bias, order[b.name]),
        )
        SESSION_STORE.put(routing_key, chosen.name)
        return chosen, "new"

    # No routing key: pick least-loaded backend with round-robin tie-breaking
    RR_COUNTER = (RR_COUNTER + 1) % len(BACKENDS)
    n = len(BACKENDS)
    order = {BACKENDS[(RR_COUNTER + i) % n].name: i for i in range(n)}
    chosen = min(candidates, key=lambda b: (b.active_requests + b.load_bias, order[b.name]))
    return chosen, "least-conn"


async def health_check_loop(app: web.Application) -> None:
    timeout = aiohttp.ClientTimeout(total=2.0, connect=1.5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            for b in BACKENDS:
                try:
                    async with session.get(f"{b.url}/health") as resp:
                        ok = resp.status == 200
                        if ok and not b.healthy:
                            logger.info("Backend %s (%s) recovered -> HEALTHY", b.name, b.url)
                        elif not ok and b.healthy:
                            logger.warning("Backend %s (%s) returned HTTP %d -> UNHEALTHY", b.name, b.url, resp.status)
                        b.healthy = ok
                        b.last_check_ts = time.time()
                        if ok:
                            b.last_error = ""
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if b.healthy:
                        logger.warning("Backend %s (%s) health check failed: %s -> UNHEALTHY", b.name, b.url, e)
                    b.healthy = False
                    b.last_error = str(e)
                    b.last_check_ts = time.time()
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)


async def handle_router_status(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "healthy_nodes": sum(1 for b in BACKENDS if b.healthy),
            "total_nodes": len(BACKENDS),
            "total_active_requests": sum(b.active_requests for b in BACKENDS),
            "sticky_sessions": len(SESSION_STORE),
            "max_active_per_node": MAX_ACTIVE_PER_NODE,
            "backends": [
                {
                    "name": b.name,
                    "url": b.url,
                    "healthy": b.healthy,
                    "active_requests": b.active_requests,
                    "load_bias": b.load_bias,
                    "total_requests": b.total_requests,
                    "failed_requests": b.failed_requests,
                    "last_error": b.last_error,
                }
                for b in BACKENDS
            ],
        }
    )


async def handle_health(request: web.Request) -> web.Response:
    if any(b.healthy for b in BACKENDS):
        return web.Response(status=200, text="")
    return web.Response(status=503, text="All backends unhealthy")


async def handle_proxy(request: web.Request) -> web.StreamResponse:
    body_bytes = await request.read()
    routing_key = extract_routing_key(request.headers, body_bytes)

    fwd_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS
    }

    client_session: aiohttp.ClientSession = request.app["client_session"]
    tried = set()
    last_exc: Optional[Exception] = None

    for attempt in range(len(BACKENDS)):
        backend, affinity = select_backend(routing_key, tried)
        tried.add(backend.name)
        target_url = f"{backend.url}{request.rel_url}"

        backend.active_requests += 1
        backend.total_requests += 1
        start_ts = time.time()
        stream_resp = None
        try:
            async with client_session.request(
                method=request.method,
                url=target_url,
                headers=fwd_headers,
                data=body_bytes if body_bytes else None,
                allow_redirects=False,
            ) as upstream_resp:
                resp_headers = {
                    k: v
                    for k, v in upstream_resp.headers.items()
                    if k.lower() not in HOP_BY_HOP_HEADERS
                }
                resp_headers["X-Routed-Backend"] = backend.name
                resp_headers["X-Cache-Affinity"] = affinity
                if routing_key:
                    resp_headers["X-Routing-Key"] = routing_key

                stream_resp = web.StreamResponse(
                    status=upstream_resp.status,
                    reason=upstream_resp.reason,
                    headers=resp_headers,
                )
                await stream_resp.prepare(request)

                async for chunk in upstream_resp.content.iter_any():
                    if chunk:
                        await stream_resp.write(chunk)

                await stream_resp.write_eof()
                elapsed = time.time() - start_ts
                logger.info(
                    "%s %s -> %s (%s, key=%s) HTTP %d in %.2fs [active=%d]",
                    request.method,
                    request.path,
                    backend.name,
                    affinity,
                    routing_key or "-",
                    upstream_resp.status,
                    elapsed,
                    backend.active_requests - 1,
                )
                return stream_resp

        except (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError, asyncio.TimeoutError) as e:
            backend.failed_requests += 1
            backend.last_error = str(e)
            last_exc = e
            logger.warning(
                "Upstream connect/read error on %s (%s) attempt %d: %s",
                backend.name,
                target_url,
                attempt + 1,
                e,
            )
            if stream_resp is not None and stream_resp.prepared:
                return stream_resp
            continue
        except (ConnectionResetError, aiohttp.ClientPayloadError) as e:
            # Client disconnected mid-stream or upstream reset mid-stream
            logger.info("Stream closed on %s (%s): %s", backend.name, request.path, e)
            return web.Response(status=499, text="Client closed request")
        finally:
            backend.active_requests = max(0, backend.active_requests - 1)

    logger.error("All backends failed for %s %s: %s", request.method, request.path, last_exc)
    return web.json_response(
        {"error": {"message": f"All vLLM backends unreachable: {last_exc}", "type": "router_error"}},
        status=502,
    )


async def on_startup(app: web.Application) -> None:
    connector = aiohttp.TCPConnector(limit=512, limit_per_host=128, keepalive_timeout=60)
    timeout = aiohttp.ClientTimeout(total=3600, connect=5.0, sock_connect=5.0, sock_read=1800.0)
    app["client_session"] = aiohttp.ClientSession(connector=connector, timeout=timeout)
    app["health_task"] = asyncio.create_task(health_check_loop(app))


async def on_cleanup(app: web.Application) -> None:
    app["health_task"].cancel()
    try:
        await app["health_task"]
    except asyncio.CancelledError:
        pass
    await app["client_session"].close()


def create_app() -> web.Application:
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/router/status", handle_router_status)
    app.router.add_get("/health", handle_health)
    os.makedirs("/home/water-float/data/videos", exist_ok=True)
    app.router.add_static("/water/videos", "/home/water-float/data/videos", show_index=False)
    app.router.add_route("*", "/{tail:.*}", handle_proxy)
    return app


if __name__ == "__main__":
    web.run_app(create_app(), host=LISTEN_HOST, port=LISTEN_PORT, access_log=None)
