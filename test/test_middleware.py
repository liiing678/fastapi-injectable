"""Tests for InjectableScopeMiddleware: request-aligned dependency lifecycles.

The middleware binds one ``InjectableScope`` to each HTTP request / WebSocket
connection, so dependencies resolved through fastapi-injectable during a request:

- are cached only for that request (never in the process-global cache),
- are unique per provider within the request across injection styles,
- run their generator cleanup exactly when the request ends.
"""

import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request, WebSocket
from fastapi.testclient import TestClient

from fastapi_injectable import (
    INJECTABLE_SCOPE_KEY,
    InjectableScope,
    InjectableScopeMiddleware,
    async_get_injected_obj,
    get_injected_obj,
    injectable,
)
from fastapi_injectable.async_exit_stack import async_exit_stack_manager
from fastapi_injectable.cache import dependency_cache
from fastapi_injectable.scope import _current_scope


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    """Poll ``predicate`` until it holds or ``timeout`` seconds elapse.

    TestClient returns as soon as the response body is complete; the middleware's
    scope teardown runs right after, on the app's own loop. Polling avoids racing it.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class Resource:
    def __init__(self) -> None:
        self.cleaned_up = False

    def cleanup(self) -> None:
        self.cleaned_up = True


def _assert_global_state_untouched() -> None:
    """Nothing resolved during a request may leak into the process-global state."""
    assert len(async_exit_stack_manager._stacks) == 0
    assert dependency_cache._shared_cache == {}
    assert len(dependency_cache._caches) == 0


def test_request_scoped_dependency_is_released_at_request_end() -> None:
    created: list[Resource] = []

    async def get_resource() -> AsyncGenerator[Resource, None]:
        resource = Resource()
        created.append(resource)
        yield resource
        resource.cleanup()

    app = FastAPI()
    app.add_middleware(InjectableScopeMiddleware)

    @app.get("/")
    async def endpoint() -> dict[str, Any]:
        resource = await async_get_injected_obj(get_resource)
        return {"id": id(resource), "cleaned_up": resource.cleaned_up}

    with TestClient(app) as client:
        first = client.get("/").json()
        second = client.get("/").json()

    # Each request gets a fresh instance: nothing is cached at application level.
    assert first["id"] != second["id"]
    # The dependency is alive for the whole request...
    assert first["cleaned_up"] is False
    assert second["cleaned_up"] is False
    # ...and released as soon as the request ends.
    assert _wait_until(lambda: len(created) == 2 and all(r.cleaned_up for r in created))
    _assert_global_state_untouched()


def test_mixed_injection_styles_share_one_instance_per_request() -> None:
    def get_mayor() -> Resource:
        return Resource()

    @injectable
    async def get_capital(mayor: Annotated[Resource, Depends(get_mayor)]) -> tuple[Resource, Resource]:
        direct = await async_get_injected_obj(get_mayor)
        return mayor, direct

    app = FastAPI()
    app.add_middleware(InjectableScopeMiddleware)

    @app.get("/")
    async def endpoint() -> dict[str, Any]:
        via_decorator, via_container = await get_capital()
        again = await async_get_injected_obj(get_mayor)
        return {
            "id": id(via_decorator),
            "unique": via_decorator is via_container is again,
        }

    with TestClient(app) as client:
        first = client.get("/").json()
        second = client.get("/").json()

    # @injectable, async_get_injected_obj and nested Depends() all see ONE instance.
    assert first["unique"] is True
    assert second["unique"] is True
    # ...but the instance does not survive the request.
    assert first["id"] != second["id"]
    _assert_global_state_untouched()


def test_sync_generator_cleanup_runs_at_request_end() -> None:
    created: list[Resource] = []

    def get_resource() -> Generator[Resource, None, None]:
        resource = Resource()
        created.append(resource)
        yield resource
        resource.cleanup()

    app = FastAPI()
    app.add_middleware(InjectableScopeMiddleware)

    @app.get("/")
    def endpoint() -> dict[str, Any]:  # sync endpoint: container resolution via get_injected_obj
        resource = get_injected_obj(get_resource)
        return {"cleaned_up": resource.cleaned_up}

    with TestClient(app) as client:
        body = client.get("/").json()

    assert body["cleaned_up"] is False
    assert _wait_until(lambda: bool(created) and created[-1].cleaned_up)
    _assert_global_state_untouched()


def test_scope_is_active_for_the_whole_request() -> None:
    app = FastAPI()
    app.add_middleware(InjectableScopeMiddleware)

    @app.get("/")
    async def endpoint(request: Request) -> dict[str, Any]:
        scope_from_asgi = request.scope[INJECTABLE_SCOPE_KEY]
        return {
            "exposed": isinstance(scope_from_asgi, InjectableScope),
            "active": _current_scope.get() is scope_from_asgi,
        }

    with TestClient(app) as client:
        body = client.get("/").json()

    assert body == {"exposed": True, "active": True}


def test_exception_in_endpoint_reaches_generator_cleanup() -> None:
    seen: list[str] = []

    async def get_resource() -> AsyncGenerator[Resource, None]:
        resource = Resource()
        try:
            yield resource
        except ValueError:
            seen.append("rollback")
            raise
        finally:
            resource.cleanup()

    app = FastAPI()
    app.add_middleware(InjectableScopeMiddleware)

    @app.get("/")
    async def endpoint() -> None:
        await async_get_injected_obj(get_resource)
        msg = "boom"
        raise ValueError(msg)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/")

    assert response.status_code == 500
    assert _wait_until(lambda: "rollback" in seen)


def test_use_cache_false_resolves_fresh_instances_but_still_cleans_up() -> None:
    created: list[Resource] = []

    async def get_resource() -> AsyncGenerator[Resource, None]:
        resource = Resource()
        created.append(resource)
        yield resource
        resource.cleanup()

    app = FastAPI()
    app.add_middleware(InjectableScopeMiddleware, use_cache=False)

    @app.get("/")
    async def endpoint() -> dict[str, Any]:
        first = await async_get_injected_obj(get_resource)
        second = await async_get_injected_obj(get_resource)
        return {"distinct": first is not second}

    with TestClient(app) as client:
        body = client.get("/").json()

    assert body["distinct"] is True
    assert _wait_until(lambda: len(created) == 2 and all(r.cleaned_up for r in created))


def test_websocket_connection_gets_its_own_scope() -> None:
    created: list[Resource] = []

    async def get_resource() -> AsyncGenerator[Resource, None]:
        resource = Resource()
        created.append(resource)
        yield resource
        resource.cleanup()

    app = FastAPI()
    app.add_middleware(InjectableScopeMiddleware)

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        resource = await async_get_injected_obj(get_resource)
        await websocket.send_json({"cleaned_up": resource.cleaned_up})
        await websocket.close()

    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json() == {"cleaned_up": False}

    assert _wait_until(lambda: bool(created) and created[-1].cleaned_up)


def test_lifespan_is_not_scoped() -> None:
    seen: list[bool] = []

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        seen.append(_current_scope.get() is None)
        yield

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(InjectableScopeMiddleware)

    @app.get("/")
    async def endpoint() -> dict[str, bool]:
        return {"ok": True}

    with TestClient(app) as client:
        assert client.get("/").json() == {"ok": True}

    assert seen == [True]
