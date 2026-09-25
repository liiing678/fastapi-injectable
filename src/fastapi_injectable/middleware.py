from __future__ import annotations

import contextvars
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send

_current_request_context: contextvars.ContextVar[RequestContext | None] = contextvars.ContextVar(
    "fastapi_injectable_request_context", default=None
)


class RequestContext:
    """The dependency lifecycle of a single HTTP request / websocket session.

    Created and torn down by :class:`InjectableRequestMiddleware` exactly once per
    request. While it is the active context, ``resolve_dependencies`` routes
    generator cleanup into this context's exit stack and caches resolved values in
    this context's cache, so container-resolved dependencies live exactly as long
    as the framework's request -- and are released when it ends, never leaking
    into the next request (or the next test).

    The real ASGI ``scope`` is kept so that resolution can happen against the
    actual request: ``Request`` dependencies see the real headers/cookies/
    ``request.app``/``request.state``, and on fastapi>=0.121 generator
    dependencies are entered into the request's own exit stacks, letting FastAPI
    itself run their cleanup.
    """

    def __init__(self, scope: Scope, *, use_cache: bool = True) -> None:
        self.scope = scope
        self.exit_stack = AsyncExitStack()
        self._use_cache = use_cache
        self._cache: dict[Any, Any] = {}

    def get_cache(self) -> dict[Any, Any] | None:
        """The request's dependency cache (``None`` when caching is disabled)."""
        if not self._use_cache:
            return None
        return self._cache


class InjectableRequestMiddleware:
    """Pure-ASGI middleware that aligns the container's scope with the request scope.

    Install it once around the FastAPI app::

        app = FastAPI()
        app.add_middleware(InjectableRequestMiddleware)
        # or: app = InjectableRequestMiddleware(app)

    For every HTTP request (and websocket session) it opens a
    :class:`RequestContext` for the duration of the request:

    - Dependencies resolved via ``@injectable``, ``get_injected_obj`` or
      ``async_get_injected_obj`` while the request is being handled are cached
      per request instead of in the process-global cache, so the same provider
      yields one instance per request and requests (and tests) never pollute
      each other.
    - Generator-style dependencies are cleaned up when the request ends. If the
      request fails, the in-flight exception is forwarded to the exit stack --
      exactly as FastAPI unwinds a failing request -- so generators receive it
      at their ``yield`` and can run their ``except``/rollback branches.
    - Resolution happens against the real request, so ``Request`` /
      ``request.app`` / ``request.state`` dependencies see actual request data.

    An explicit ``injectable_scope()`` entered inside the handler still takes
    precedence over the request context. Without the middleware nothing changes:
    resolution falls back to the process-global cache and exit stacks.

    Note:
        FastAPI's own ``Depends`` resolution keeps its private per-request cache,
        so a provider used both natively in a route signature and through the
        container is instantiated once per side. Use one injection style per
        provider if a single shared instance is required.
    """

    def __init__(self, app: ASGIApp, *, use_cache: bool = True) -> None:
        self.app = app
        self.use_cache = use_cache

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        context = RequestContext(scope, use_cache=self.use_cache)
        token = _current_request_context.set(context)
        try:
            await context.exit_stack.__aenter__()
            try:
                await self.app(scope, receive, send)
            except BaseException as exc:
                # Unwind with the in-flight exception -- exactly as FastAPI unwinds a
                # failing request -- so generator dependencies receive it at their
                # ``yield`` and can run ``except``/rollback branches (issue #255).
                # The original exception always keeps propagating afterwards.
                await context.exit_stack.__aexit__(type(exc), exc, exc.__traceback__)
                raise
            else:
                await context.exit_stack.__aexit__(None, None, None)
        finally:
            _current_request_context.reset(token)
