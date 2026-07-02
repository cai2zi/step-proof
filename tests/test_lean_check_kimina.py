from __future__ import annotations

from proofflow.lean_check import _analyze_kimina_repl_response


def test_kimina_valid_response_is_verified() -> None:
    lean_pass, lean_verify, output = _analyze_kimina_repl_response(
        {"id": "valid", "response": {"env": 0}}
    )

    assert lean_pass is True
    assert lean_verify is True
    assert output is None


def test_kimina_lean_error_fails_pass_and_verify() -> None:
    message = {
        "severity": "error",
        "pos": {"line": 3, "column": 0},
        "endPos": {"line": 3, "column": 5},
        "data": "unsolved goals",
    }
    lean_pass, lean_verify, output = _analyze_kimina_repl_response(
        {"id": "lean-error", "response": {"env": 0, "messages": [message]}}
    )

    assert lean_pass is False
    assert lean_verify is False
    assert output == {"messages": [message]}


def test_kimina_sorry_passes_but_does_not_verify() -> None:
    sorry = {
        "pos": {"line": 4, "column": 0},
        "endPos": {"line": 4, "column": 5},
    }
    lean_pass, lean_verify, output = _analyze_kimina_repl_response(
        {"id": "sorry", "response": {"env": 0, "sorries": [sorry]}}
    )

    assert lean_pass is True
    assert lean_verify is False
    assert output == {"sorries": [sorry]}


def test_kimina_has_sorry_marker_passes_but_does_not_verify() -> None:
    lean_pass, lean_verify, output = _analyze_kimina_repl_response(
        {
            "id": "has-sorry",
            "response": {
                "env": 0,
                "messages": [
                    {
                        "severity": "warning",
                        "data": "declaration uses 'sorry'",
                    }
                ],
            },
        }
    )

    assert lean_pass is True
    assert lean_verify is False
    assert output["messages"][0]["data"] == "declaration uses 'sorry'"


def test_kimina_server_error_fails_pass_and_verify() -> None:
    lean_pass, lean_verify, output = _analyze_kimina_repl_response(
        {"id": "timeout", "error": "Lean REPL command timed out in 300 seconds"}
    )

    assert lean_pass is False
    assert lean_verify is False
    assert output == "Lean REPL command timed out in 300 seconds"
