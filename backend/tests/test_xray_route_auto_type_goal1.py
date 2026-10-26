"""Goal 1 tests for region-free automatic X-ray upload."""

import ast
from pathlib import Path


ROUTE_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "routes"
    / "xray.py"
)


def _route_tree():
    """Read and parse the current X-ray route without importing the app."""
    source = ROUTE_PATH.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _upload_xray_function(tree):
    """Find the async upload_xray endpoint."""
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "upload_xray":
            return node

    raise AssertionError("upload_xray endpoint function was not found")


def test_upload_endpoint_does_not_require_manual_region():
    """A user-selected region must not be a required endpoint argument."""
    _, tree = _route_tree()
    function = _upload_xray_function(tree)

    argument_names = [
        argument.arg
        for argument in (
            function.args.posonlyargs + function.args.args
        )
    ]

    assert "region" not in argument_names
    assert "body_region" not in argument_names


def test_upload_endpoint_routes_from_image_only():
    """The stored image path is passed to automatic routing."""
    _, tree = _route_tree()
    function = _upload_xray_function(tree)

    route_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "route_xray"
    ]

    assert route_calls, "upload_xray must invoke route_xray"

    assert any(
        len(call.args) == 1 and not call.keywords
        for call in route_calls
    ), "route_xray should receive the image path without a manual region"


def test_upload_response_exposes_automatic_detection():
    """The frontend needs the automatic classifier's response metadata."""
    source, _ = _route_tree()

    required_fields = (
        '"image_type_detection"',
        '"detected_region"',
        '"region_source"',
        '"anatomy_verification_status"',
        '"body_part_confidence"',
        '"report_region_verification"',
    )

    for field in required_fields:
        assert field in source, f"Missing response field: {field}"


def test_route_never_uses_filename_for_anatomy_detection():
    """Routing must use model output rather than filename keywords."""
    source, _ = _route_tree()

    assert "route_xray(str(dest))" in source
    assert "Path(file.filename" in source  # Filename is used for safe storage.
    assert "infer_region_from_filename" not in source