import asyncio
from collections.abc import Generator
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, Request, WebSocket
from fastapi.testclient import TestClient

from fastapi_injectable import (
    InjectableRequestMiddleware,
    async_get_injected_obj,
    injectable,
    injectable_scope,
)
from fastapi_injectable.cache import dependency_cache
from fastapi_injectable.middleware import _current_request_context


def _build_app(*, use_cache: bool = True) -> FastAPI:
    app = FastAPI()
    app.add_middleware(InjectableRequestMiddleware, use_cache=use_cache)
    return app


async def test_request_context_is_set_during_request_and_reset_afterwards() -> None:
    app = _build_app()
    seen: list[Any] = []

    @app.get("/")
    async def route() -> dict[str, bool]:
        context = _current_request_context.get()
        seen.append(context)
        return {"in_request": context is not None}

    with TestClient(app) as client:
        assert client.get("/").json() == {"in_request": True}
        assert client.get("/").json() == {"in_request": True}

    assert _current_request_context.get() is None
    # Each request got its own context.
    assert len(seen) == 2
    assert seen[0] is not seen[1]


async def test_dependency_instance_is_unique_per_request_not_cached_app_wide() -> None:
    app = _build_app()

    def get_object() -> object:
        return object()

    @app.get("/")
    async def route() -> dict[str, Any]:
        first = await async_get_injected_obj(get_object)
        second = await async_get_injected_obj(get_object)
        return {"same_within_request": first is second, "id": id(first)}

    with TestClient(app) as client:
        resp1 = client.get("/").json()
        resp2 = client.get("/").json()

    assert resp1["same_within_request"] is True
    assert resp2["same_within_request"] is True
    # A request-scoped dependency must not be cached at application level.
    assert resp1["id"] != resp2["id"]
    assert dependency_cache.get() == {}


async def test_decorator_and_functional_api_share_one_instance_per_request() -> None:
    app = _build_app()

    def get_service() -> object:
        return object()

    @injectable
    async def use_service(service: Annotated[object, Depends(get_service)]) -> object:
        return service

    @app.get("/")
    async def route() -> dict[str, bool]:
        via_decorator = await use_service()
        via_functional = await async_get_injected_obj(get_service)
        return {"same": via_decorator is via_functional}

    with TestClient(app) as client:
        assert client.get("/").json() == {"same": True}


async def test_generator_cleanup_runs_at_request_end() -> None:
    app = _build_app()
    events: list[str] = []

    def get_resource() -> Generator[object, None, None]:
        resource = object()
        events.append("setup")
        yield resource
        events.append("teardown")

    @app.get("/")
    async def route() -> dict[str, Any]:
        resource = await async_get_injected_obj(get_resource)
        return {"resource_id": id(resource), "events_during_request": list(events)}

    with TestClient(app) as client:
        resp1 = client.get("/").json()
        # Cleanup ran exactly once, after the endpoint finished.
        assert resp1["events_during_request"] == ["setup"]
        assert events == ["setup", "teardown"]

        resp2 = client.get("/").json()
        assert events == ["setup", "teardown", "setup", "teardown"]

    # A finished request's resource is never reused by the next one.
    assert resp1["resource_id"] != resp2["resource_id"]


async def test_generator_receives_exception_when_request_fails() -> None:
    app = _build_app()
    seen_exceptions: list[str] = []

    def get_resource() -> Generator[object, None, None]:
        try:
            yield object()
        except ValueError:
            seen_exceptions.append("rollback")
            raise

    @app.get("/")
    async def route() -> None:
        await async_get_injected_obj(get_resource)
        msg = "boom"
        raise ValueError(msg)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/")

    assert response.status_code == 500
    assert seen_exceptions == ["rollback"]


async def test_request_dependency_resolves_to_the_real_request() -> None:
    app = _build_app()

    def get_marker(request: Request) -> dict[str, Any]:
        return {
            "marker": request.headers.get("x-marker"),
            "app_is_fastapi_app": request.app is app,
        }

    @app.get("/")
    async def route() -> dict[str, Any]:
        return await async_get_injected_obj(get_marker)

    with TestClient(app) as client:
        response = client.get("/", headers={"x-marker": "real"})

    assert response.json() == {"marker": "real", "app_is_fastapi_app": True}


async def test_explicit_injectable_scope_takes_precedence_over_request_context() -> None:
    app = _build_app()
    events: list[str] = []

    def get_object() -> object:
        return object()

    def get_resource() -> Generator[object, None, None]:
        events.append("setup")
        yield object()
        events.append("teardown")

    @app.get("/")
    async def route() -> dict[str, Any]:
        request_wide = await async_get_injected_obj(get_object)
        async with injectable_scope():
            scoped = await async_get_injected_obj(get_object)
            await async_get_injected_obj(get_resource)
        events_after_scope_exit = list(events)
        request_wide_again = await async_get_injected_obj(get_object)
        return {
            "scope_isolated_from_request": scoped is not request_wide,
            "request_cache_preserved": request_wide_again is request_wide,
            "events_after_scope_exit": events_after_scope_exit,
        }

    with TestClient(app) as client:
        body = client.get("/").json()

    assert body["scope_isolated_from_request"] is True
    assert body["request_cache_preserved"] is True
    # The explicit scope cleaned up its own generator at scope exit, mid-request.
    assert body["events_after_scope_exit"] == ["setup", "teardown"]


async def test_middleware_use_cache_false_disables_request_cache() -> None:
    app = _build_app(use_cache=False)

    def get_object() -> object:
        return object()

    @app.get("/")
    async def route() -> dict[str, bool]:
        first = await async_get_injected_obj(get_object)
        second = await async_get_injected_obj(get_object)
        return {"same": first is second}

    with TestClient(app) as client:
        assert client.get("/").json() == {"same": False}


async def test_concurrent_requests_are_isolated() -> None:
    app = _build_app()

    def get_object() -> object:
        return object()

    @app.get("/")
    async def route() -> dict[str, int]:
        obj = await async_get_injected_obj(get_object)
        await asyncio.sleep(0.01)  # force interleaving of the two requests
        # Still the same instance after other requests had a chance to run.
        again = await async_get_injected_obj(get_object)
        assert again is obj
        return {"id": id(obj)}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp1, resp2 = await asyncio.gather(client.get("/"), client.get("/"))

    assert resp1.json()["id"] != resp2.json()["id"]


async def test_websocket_session_gets_request_scoped_lifecycle() -> None:
    app = _build_app()
    events: list[str] = []

    def get_resource() -> Generator[object, None, None]:
        events.append("setup")
        yield object()
        events.append("teardown")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        resource = await async_get_injected_obj(get_resource)
        await websocket.send_json({"events": list(events), "resource": id(resource)})
        await websocket.close()

    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["events"] == ["setup"]

    assert events == ["setup", "teardown"]


async def test_lifespan_passes_through_without_request_context() -> None:
    app = _build_app()
    observed: list[Any] = []

    @app.on_event("startup")
    async def startup() -> None:
        observed.append(_current_request_context.get())

    with TestClient(app):
        pass

    assert observed == [None]


async def test_without_middleware_global_behavior_is_unchanged() -> None:
    # Compatibility guard: an app without the middleware keeps the historical
    # process-global caching behavior (the pytest plugin resets it between tests).
    app = FastAPI()

    def get_object() -> object:
        return object()

    @app.get("/")
    async def route() -> dict[str, int]:
        obj = await async_get_injected_obj(get_object)
        return {"id": id(obj)}

    with TestClient(app) as client:
        resp1 = client.get("/").json()
        resp2 = client.get("/").json()

    assert resp1["id"] == resp2["id"]
