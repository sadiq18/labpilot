"""An OpenAI-compatible front door for the gateway.

External tools speak HTTP; `LLMGateway` is a Python API. Without this, anything
that is not labpilot — `aider` first — would call providers directly and take
its own routing, budget, failover and provenance with it. Measured cost of that:
the ledger never sees the tokens, rate limiting counts several calls as one, and
`cool_down` + re-select never fires for the most important call in the system.

So the tool becomes a *client* of the router rather than a peer.

**Roles survive the boundary.** HTTP carries a model name, not a role, so a
request names ``labpilot/<role>`` and this module resolves it through the same
`select_route` every internal caller uses. Per-role `requires`, `on_exhaustion`
and `requires_strong` keep working; the caller still never names a vendor, which
is the property M10 exists to protect.

**Why `http.server`.** `fitroute` is stdlib + pydantic only, enforced by
`test_fitroute_uses_only_the_standard_library_and_pydantic`, so that extraction
stays a directory move. The surface here is one POST and one GET, called by a
single local subprocess: a framework would buy validation (pydantic, already
here), routing (one route), async (the gateway is synchronous) and OpenAPI docs
(irrelevant), in exchange for starlette, anyio and uvicorn. If this ever becomes
a hosted multi-tenant service it should be a real ASGI app living outside
`fitroute` — see the design's rollout notes.

**Non-streaming only.** `stream: true` is refused rather than silently ignored.
Streaming omits the `usage` field unless the upstream honours
``stream_options.include_usage``, and an unmetered call is exactly what the
ledger cannot afford; failover also cannot survive the first byte. Answering a
streaming request with a complete body would look like success while quietly
disabling the accounting this server exists to provide.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from fitroute.gateway import LLMGateway, RoleUnavailable

logger = logging.getLogger(__name__)

#: Model names this server answers to. The ``labpilot/`` prefix is **required**.
#:
#: An earlier version accepted a bare name too, "because litellm sometimes
#: strips prefixes" — which made `gpt-4o` a valid *role*. `role_spec` then falls
#: through to `default` and the request routes with no role at all: exactly the
#: bypass this function exists to prevent, introduced by being lenient about the
#: one token that carries the meaning.
#:
#: If a client really does strip the prefix, that is a client to configure, not
#: a guard to loosen.
_ROLE_MODEL = re.compile(r"^labpilot/(?P<role>[a-z0-9_-]+)$", re.IGNORECASE)

#: How long a client should wait when nothing is routable. Bounded so a
#: misconfigured client cannot be told to sleep for an hour.
_MAX_RETRY_AFTER = 300


class ProxyError(Exception):
    """A request that cannot be served, with the status the client should see."""

    def __init__(self, status: HTTPStatus, message: str, *, retry_after: int | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.retry_after = retry_after


def role_from_model(model: str) -> str:
    """``labpilot/codegen`` -> ``codegen``.

    Rejects anything else rather than guessing. A request naming a real provider
    model would otherwise silently bypass role selection — the failure this
    server exists to prevent, arriving through the server itself.
    """
    match = _ROLE_MODEL.match((model or "").strip())
    if not match:
        raise ProxyError(
            HTTPStatus.BAD_REQUEST,
            f"model must be 'labpilot/<role>', got {model!r}. This proxy routes "
            "by role; naming a provider model would bypass routing entirely.",
        )
    return match.group("role").lower()


def split_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """Flatten chat messages into the ``(system, user)`` the gateway takes.

    The gateway's contract is one system and one user string. Multi-turn
    histories are joined rather than truncated: aider sends the whole
    conversation, and dropping earlier turns would silently change the request.
    """
    system_parts: list[str] = []
    user_parts: list[str] = []
    for message in messages or []:
        content = message.get("content")
        if isinstance(content, list):
            # OpenAI content-parts form: keep the text, ignore images.
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        text = str(content or "")
        if not text:
            continue
        if str(message.get("role") or "").lower() == "system":
            system_parts.append(text)
        else:
            user_parts.append(text)
    return "\n\n".join(system_parts), "\n\n".join(user_parts)


def completion_response(text: str, model: str, served: Any) -> dict[str, Any]:
    """An OpenAI-shaped chat completion.

    `usage` reports what the ledger actually recorded. Clients estimate cost
    from it, so inventing numbers here would make their accounting disagree with
    ours in a way neither side could detect.
    """
    tokens = int(getattr(served, "tokens", 0) or 0)
    return {
        "id": f"chatcmpl-fitroute-{id(text):x}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": tokens, "total_tokens": tokens},
        # Non-standard, and deliberately visible: which provider actually served
        # this. A client logging the response gets the same attribution the
        # evidence card does.
        "x_fitroute": {
            "provider": getattr(served, "provider", None),
            "model": getattr(served, "model", None),
            "role": getattr(served, "role", None),
            "degraded": bool(getattr(served, "degraded", False)),
            "cache_hit": bool(getattr(served, "cache_hit", False)),
        },
    }


def handle_chat_completion(gateway: LLMGateway, payload: dict[str, Any]) -> dict[str, Any]:
    """Route one chat-completion request. Raises `ProxyError` on refusal."""
    if payload.get("stream"):
        raise ProxyError(
            HTTPStatus.BAD_REQUEST,
            "streaming is not supported: the usage field it omits is what the "
            "budget ledger records, and failover cannot survive the first byte. "
            "Pass --no-stream.",
        )

    role = role_from_model(str(payload.get("model") or ""))
    system, user = split_messages(payload.get("messages") or [])
    if not user:
        raise ProxyError(HTTPStatus.BAD_REQUEST, "no user message content")

    client = gateway.for_role(role)
    json_mode = str((payload.get("response_format") or {}).get("type") or "") == "json_object"
    try:
        # `allow_wait=False` is the whole point. In-process callers may pace on
        # `wait_seconds`; this one is holding aider's socket, and the call runs
        # inside `_GATEWAY_LOCK` — so a paced wait would block every other
        # proxied request behind it, for up to `max_wait_seconds` (900s for
        # codegen, whose `on_exhaustion` is `wait`). Refusing fast and letting
        # the client back off is both faster and honest.
        text = client.complete(system, user, json_mode=json_mode, allow_wait=False)
    except RoleUnavailable as exc:
        # Genuine exhaustion: `select_route` tried every provider and, for
        # `degrade` roles, every weaker one. 429 + Retry-After is the answer
        # litellm honours natively, and it reports the wait the router actually
        # computed rather than a constant — a client told to back off for the
        # real window retries at the right time.
        wait = getattr(exc, "retry_after", None)
        raise ProxyError(
            HTTPStatus.TOO_MANY_REQUESTS,
            str(exc),
            retry_after=min(int(wait), _MAX_RETRY_AFTER) if wait else _MAX_RETRY_AFTER,
        ) from exc
    return completion_response(text, str(payload.get("model")), client.last_served)


#: Roles that always appear in `/v1/models`, whether or not a workspace names
#: them. A role with no explicit entry still routes — `role_spec` falls back to
#: `default` — but a client that probes this list first may refuse a model id it
#: has not seen, so omitting them would break `labpilot/codegen` on exactly the
#: workspaces that did not bother to configure it.
_WELL_KNOWN_ROLES = ("default", "codegen", "reasoning", "summarize")


def handle_models(gateway: LLMGateway) -> dict[str, Any]:
    """Advertise roles, not providers — litellm probes this on startup."""
    roles = sorted(set(gateway.routing.roles) | set(_WELL_KNOWN_ROLES))
    return {
        "object": "list",
        "data": [
            {"id": f"labpilot/{role}", "object": "model", "owned_by": "fitroute"}
            for role in roles
        ],
    }


#: Serialises gateway access across handler threads.
#:
#: `ThreadingHTTPServer` dispatches each request on a worker thread, but
#: `BudgetLedger` holds a sqlite connection bound to the thread that opened it —
#: so every real request failed with "SQLite objects created in a thread can
#: only be used in that same thread". Serialising is the honest fix at this
#: scale: the proxy serves one local subprocess, the gateway call is the slow
#: part anyway, and a lock keeps the ledger's read-modify-write consistent,
#: which matters more here than concurrency. Threads are kept so a slow upstream
#: cannot wedge the accept loop.
_GATEWAY_LOCK = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    gateway: LLMGateway  # set on the subclass by `build_handler`

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        logger.debug("proxy %s", fmt % args)

    def _send(
        self,
        status: HTTPStatus,
        body: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, exc: ProxyError) -> None:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else {}
        self._send(
            exc.status,
            {"error": {"message": exc.message, "type": "fitroute_error", "code": exc.status}},
            headers,
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if self.path.rstrip("/").endswith("/models"):
            self._send(HTTPStatus.OK, handle_models(self.gateway))
            return
        self._error(ProxyError(HTTPStatus.NOT_FOUND, f"no route for {self.path}"))

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._error(ProxyError(HTTPStatus.NOT_FOUND, f"no route for {self.path}"))
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(ProxyError(HTTPStatus.BAD_REQUEST, f"invalid JSON body: {exc}"))
            return
        try:
            with _GATEWAY_LOCK:
                response = handle_chat_completion(self.gateway, payload)
            self._send(HTTPStatus.OK, response)
        except ProxyError as exc:
            self._error(exc)
        except Exception as exc:  # noqa: BLE001
            # An upstream failure the gateway could not route around. 502 rather
            # than 500: the fault is upstream, and clients back off differently.
            logger.warning("proxy upstream failure: %s", exc)
            self._error(ProxyError(HTTPStatus.BAD_GATEWAY, str(exc)))


def build_handler(gateway: LLMGateway) -> type[_Handler]:
    return type("FitrouteHandler", (_Handler,), {"gateway": gateway})


class ProxyServer:
    """A localhost proxy, scoped to whoever started it.

    Deliberately not a daemon. A long-running server needs its own supervision,
    concurrent campaigns would collide on a fixed port, and an orphan outlives
    the run whose budget ledger it is writing to. Binding to port 0 and stopping
    in the owner's ``finally`` means it cannot outlive what it accounts for.
    """

    def __init__(self, gateway: LLMGateway, *, host: str = "127.0.0.1", port: int = 0) -> None:
        # There is no authentication. Bound to loopback that is fine — the only
        # client is a subprocess this process started. Binding elsewhere turns
        # it into an unauthenticated spend path against the operator's keys, so
        # a non-loopback host says so out loud rather than failing quietly the
        # first time someone copies the line.
        if host not in ("127.0.0.1", "localhost", "::1"):
            logger.warning(
                "fitroute proxy binding to %s: there is no auth on this server, "
                "so anything that can reach it can spend the configured API keys",
                host,
            )
        self._server = ThreadingHTTPServer((host, port), build_handler(gateway))
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        """What a client passes as its OpenAI base URL."""
        host, port = self._server.server_address[0], self.port
        return f"http://{host}:{port}/v1"

    def start(self) -> ProxyServer:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("fitroute proxy listening on %s", self.base_url)
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> ProxyServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
