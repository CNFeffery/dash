"""Flask route setup, Streamable HTTP transport, and MCP message handling."""

# pylint: disable=cyclic-import
# The MCP server imports dash primitives to dispatch callbacks, and dash
# lazy-imports this module to wire the MCP endpoint. Cycle is managed here.

from __future__ import annotations

import json
import logging
from functools import reduce
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

from flask import Response, request
from mcp.types import (
    LATEST_PROTOCOL_VERSION,
    ErrorData,
    Implementation,
    InitializeResult,
    JSONRPCError,
    JSONRPCResponse,
    ResourcesCapability,
    ServerCapabilities,
    ToolsCapability,
)

from dash import get_app
from dash._get_app import with_app_context_factory
from dash.mcp.primitives import (
    call_tool,
    list_resource_templates,
    list_resources,
    list_tools,
    read_resource,
)
from dash.mcp.tasks import get_task, get_task_result, cancel_task
from dash.mcp.primitives.tools.callback_adapter_collection import (
    CallbackAdapterCollection,
)
from dash.mcp.types import MCPError
from dash.version import __version__

if TYPE_CHECKING:
    from dash import Dash

logger = logging.getLogger(__name__)


def _url_from_path(*parts: str) -> str:
    """Build an absolute URL by joining path parts onto the current request origin.

    Behind a reverse proxy, TLS terminates at the proxy so
    ``request.scheme`` reports HTTP even when the client connected
    over HTTPS.  Use HTTPS unless running on localhost.
    """
    host = request.host
    is_localhost = host.startswith("localhost") or host.startswith("127.0.0.1")
    scheme = "http" if is_localhost else "https"
    path = reduce(urljoin, parts, "/")
    return f"{scheme}://{host}{path}"


def _setup_mcp_oauth(app: Dash, mcp_path: str, mcp_authorization_server: str) -> None:
    """Register OAuth metadata endpoint and auth gate for MCP.

    Serves RFC 9728 Protected Resource Metadata so MCP clients can
    discover the authorization server, and returns 401 with
    WWW-Authenticate for unauthenticated requests to the MCP endpoint.
    """
    well_known_path = urljoin("/.well-known/oauth-protected-resource/", mcp_path)

    def _serve_resource_metadata() -> Response:
        return Response(
            json.dumps(
                {
                    "resource": _url_from_path(
                        app.config.requests_pathname_prefix, mcp_path
                    ),
                    "authorization_servers": [mcp_authorization_server],
                    "bearer_methods_supported": ["header"],
                }
            ),
            content_type="application/json",
        )

    app._add_url(well_known_path.lstrip("/"), _serve_resource_metadata)

    @app.server.before_request
    def _mcp_require_auth():
        if request.path != app.config.routes_pathname_prefix + mcp_path:
            return None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return None
        resource_metadata_url = _url_from_path(well_known_path)
        return Response(
            json.dumps({"error": "unauthorized"}),
            status=401,
            content_type="application/json",
            headers={
                "WWW-Authenticate": (
                    f'Bearer resource_metadata="{resource_metadata_url}"'
                ),
            },
        )

    logger.info("MCP OAuth enabled, authorization server: %s", mcp_authorization_server)


def enable_mcp_server(
    app: Dash,
    mcp_path: str,
    mcp_authorization_server: str | None = None,
) -> None:
    """Add MCP routes to a Dash/Flask app."""
    # -- Streamable HTTP endpoint --------------------------------------------

    def mcp_handler() -> Response:
        if request.method == "POST":
            return _handle_post()
        if request.method == "GET":
            return _handle_get()
        if request.method == "DELETE":
            return _handle_delete()
        return Response(
            json.dumps({"error": "Method not allowed"}),
            content_type="application/json",
            status=405,
        )

    def _handle_get() -> Response:
        # MCP spec allows servers to opt out of GET-initiated SSE streams
        # by returning 405. We don't push server-initiated events.
        return Response(
            json.dumps({"error": "Method not allowed"}),
            content_type="application/json",
            status=405,
        )

    def _handle_post() -> Response:
        content_type = request.content_type or ""
        if "application/json" not in content_type:
            return Response(
                json.dumps({"error": "Content-Type must be application/json"}),
                content_type="application/json",
                status=415,
            )

        data = request.get_json(silent=True)
        if data is None:
            return Response(
                json.dumps({"error": "Invalid JSON"}),
                content_type="application/json",
                status=400,
            )

        response_data = _process_mcp_message(data)

        if response_data is None:
            return Response("", status=202)

        return Response(
            json.dumps(response_data),
            content_type="application/json",
            status=200,
        )

    def _handle_delete() -> Response:
        # No sessions to terminate — server is stateless.
        return Response(
            json.dumps({"error": "Method not allowed"}),
            content_type="application/json",
            status=405,
        )

    # -- Register routes -----------------------------------------------------

    # pylint: disable-next=protected-access
    app._add_url(
        mcp_path, with_app_context_factory(mcp_handler, app), ["GET", "POST", "DELETE"]
    )

    if mcp_authorization_server:
        _setup_mcp_oauth(app, mcp_path, mcp_authorization_server)

    logger.info(
        "MCP routes registered at %s%s",
        app.config.routes_pathname_prefix,
        mcp_path,
    )


def _handle_initialize() -> InitializeResult:
    return InitializeResult(
        protocolVersion=LATEST_PROTOCOL_VERSION,
        capabilities=ServerCapabilities(
            tools=ToolsCapability(listChanged=False),
            resources=ResourcesCapability(),
        ),
        serverInfo=Implementation(name="Plotly Dash", version=__version__),
        instructions=(
            "This is a Dash web application. "
            "Dash apps are stateless: calling a tool executes "
            "a callback and returns its result to you, but does "
            "NOT update the user's browser. "
            "Use tool results to answer questions about what "
            "the app would produce for given inputs."
        ),
    )


def _process_mcp_message(data: dict[str, Any]) -> dict[str, Any] | None:
    """
    Process an MCP JSON-RPC message and return the response dict.

    Returns ``None`` for notifications (no ``id`` field).
    """
    method = data.get("method", "")
    params = data.get("params", {}) or {}
    _id = data.get("id")
    request_id: str | int = _id if isinstance(_id, (str, int)) else ""

    app = get_app()
    if not hasattr(app, "mcp_callback_map"):
        app.mcp_callback_map = CallbackAdapterCollection(app)

    mcp_methods = {
        "initialize": _handle_initialize,
        "tools/list": list_tools,
        "tools/call": lambda: call_tool(
            tool_name=params.get("name", ""),
            arguments=params.get("arguments", {}),
            task=params.get("task"),
        ),
        "resources/list": list_resources,
        "resources/templates/list": list_resource_templates,
        "resources/read": lambda: read_resource(params.get("uri", "")),
        "tasks/get": lambda: get_task(task_id=params.get("taskId", "")),
        "tasks/result": lambda: get_task_result(task_id=params.get("taskId", "")),
        "tasks/cancel": lambda: cancel_task(task_id=params.get("taskId", "")),
    }

    try:
        handler = mcp_methods.get(method)
        if handler is None:
            if method.startswith("notifications/"):
                return None
            raise ValueError(f"Unknown method: {method}")

        result = handler()

        response = JSONRPCResponse(
            jsonrpc="2.0",
            id=request_id,
            result=result.model_dump(exclude_none=True, mode="json"),
        )
        return response.model_dump(exclude_none=True, mode="json")

    except MCPError as e:
        logger.error("MCP error: %s", e)
        return JSONRPCError(
            jsonrpc="2.0",
            id=request_id,
            error=ErrorData(code=e.code, message=str(e)),
        ).model_dump(exclude_none=True)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("MCP error: %s", e, exc_info=True)
        return JSONRPCError(
            jsonrpc="2.0",
            id=request_id,
            error=ErrorData(code=-32603, message=f"{type(e).__name__}: {e}"),
        ).model_dump(exclude_none=True)
