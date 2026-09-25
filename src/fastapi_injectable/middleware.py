"""ASGI middleware that aligns fastapi-injectable's dependency lifecycle with the web request.

Without this middleware, dependencies resolved through fastapi-injectable inside a
FastAPI app are cached in the process-global dependency cache and their generator
cleanups are parked in the process-global exit-stack manager -- both application-level
lifetimes. A dependency that is conceptually request-scoped therefore leaks across
requests (and across tests that share the process).

:class:`InjectableScopeMiddleware` opens one :class:`~fastapi_injectable.scope.InjectableScope`
per HTTP request (and per WebSocket connection) and makes it the active scope for the
entire downstream handling, so the container side of the app lives and dies with the
framework's request.
"""

from starlette.types import ASGIApp, Receive, Scope, Send

from .scope import InjectableScope

#: ASGI scope key under which the request's active ``InjectableScope`` is stored,
#: mirroring FastAPI's own ``fastapi_inner_astack`` convention. Endpoints can read
#: ``request.scope[INJECTABLE_SCOPE_KEY]`` to introspect the scope serving them.
INJECTABLE_SCOPE_KEY = "fastapi_injectable_scope"

# ASGI connection types that get a scope of their own. ``lifespan`` (and any other
# scope type) passes through untouched: application-level dependencies keep using the
# process-global cache and exit-stack manager, exactly as before.
_SCOPED_ASGI_TYPES = ("http", "websocket")


class InjectableScopeMiddleware:
    """Pure-ASGI middleware that binds one ``InjectableScope`` to each request.

    For every HTTP request / WebSocket connection the middleware:

    - opens a fresh :class:`~fastapi_injectable.scope.InjectableScope` and activates it,
      so **every** fastapi-injectable resolution during the request -- ``@injectable``
      callables, ``get_injected_obj()``, ``async_get_injected_obj()``, and any
      ``Depends()`` nested inside them -- shares the scope's private cache. The same
      provider therefore yields **one instance per request** no matter which injection
      style asks for it, and never touches the process-global cache.
    - closes the scope when the request finishes (after the response is fully sent),
      running every generator dependency's cleanup exactly at request end. If the
      endpoint raised, the exception is forwarded into the generators at their
      ``yield`` -- the same rollback semantics FastAPI gives native dependencies.

    Nothing about the public API changes: ``@injectable`` and ``get_injected_obj()``
    are used exactly as before, inside or outside a request.

    Usage::

        from fastapi import FastAPI
        from fastapi_injectable import InjectableScopeMiddleware

        app = FastAPI()
        app.add_middleware(InjectableScopeMiddleware)

    Notes:
        - Dependencies declared directly in route signatures (``def endpoint(x =
          Depends(...))``) are resolved by FastAPI's own internal per-request cache,
          which FastAPI does not expose; the middleware aligns everything resolved
          *through fastapi-injectable*. To share one instance between a route
          signature and the container, resolve it through fastapi-injectable in both
          places.
        - The ``lifespan`` scope is passed through unchanged, so application-level
          dependencies keep their existing process-global lifetime.
        - Inside sync endpoints prefer ``async_get_injected_obj``-style resolution
          paths where possible; like ``InjectableScope`` itself, the scope's cache is
          partitioned per event loop (see its docstring for the rationale).
    """

    def __init__(self, app: ASGIApp, *, use_cache: bool = True) -> None:
        """Initialize the middleware.

        Args:
            app: The ASGI app to wrap (supplied by ``app.add_middleware``).
            use_cache: Whether resolutions within a request share the scope's cache.
                Defaults to True. With False, every resolution is fresh, but cleanup
                still runs at request end.
        """
        self.app = app
        self.use_cache = use_cache

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in _SCOPED_ASGI_TYPES:
            await self.app(scope, receive, send)
            return
        # ``async with`` forwards any in-flight exception to the scope's exit stack,
        # so generator dependencies see it at their ``yield`` and can roll back --
        # mirroring how FastAPI unwinds a failing request.
        async with InjectableScope(use_cache=self.use_cache) as request_scope:
            scope[INJECTABLE_SCOPE_KEY] = request_scope
            await self.app(scope, receive, send)
