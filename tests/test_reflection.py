"""T1 — Unit tests: Reflector + ReflectionResult"""

from unittest.mock import MagicMock, patch
import pytest

from agent.reflection import (
    Reflector,
    ReflectionResult,
    LowConfidenceError,
    CONFIDENCE_GATE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    return MagicMock()


@pytest.fixture
def reflector(client):
    return Reflector(client=client, agent_model="claude-opus-4-6")


def _mock_parse_response(result: ReflectionResult):
    """Build a mock object mimicking client.messages.parse() return value."""
    resp = MagicMock()
    resp.parsed_output = result
    return resp


# ── ReflectionResult schema ────────────────────────────────────────────────────

def test_reflection_result_valid():
    r = ReflectionResult(
        root_cause="File not found",
        self_criticism="Assumed wrong path",
        new_strategy=["List directory first", "Use absolute path"],
        confidence=75,
    )
    assert r.confidence == 75
    assert len(r.new_strategy) == 2


def test_reflection_result_confidence_bounds():
    with pytest.raises(Exception):
        ReflectionResult(
            root_cause="x", self_criticism="y", new_strategy=[], confidence=101
        )
    with pytest.raises(Exception):
        ReflectionResult(
            root_cause="x", self_criticism="y", new_strategy=[], confidence=-1
        )


def test_reflection_result_confidence_edge_values():
    r0 = ReflectionResult(root_cause="x", self_criticism="y", new_strategy=[], confidence=0)
    r100 = ReflectionResult(root_cause="x", self_criticism="y", new_strategy=[], confidence=100)
    assert r0.confidence == 0
    assert r100.confidence == 100


# ── _pick_depth ────────────────────────────────────────────────────────────────

def test_pick_depth_attempt_1_uses_haiku(reflector):
    model, kwargs = reflector._pick_depth(1)
    assert model == Reflector.HAIKU
    assert kwargs == {}


def test_pick_depth_attempt_2_uses_agent_model(reflector):
    model, kwargs = reflector._pick_depth(2)
    assert model == "claude-opus-4-6"
    assert kwargs == {}


def test_pick_depth_attempt_3_uses_adaptive_thinking(reflector):
    model, kwargs = reflector._pick_depth(3)
    assert model == "claude-opus-4-6"
    assert kwargs == {"thinking": {"type": "adaptive"}}


def test_pick_depth_attempt_gt3_uses_adaptive_thinking(reflector):
    model, kwargs = reflector._pick_depth(99)
    assert model == "claude-opus-4-6"
    assert "thinking" in kwargs


# ── reflect() — success path ───────────────────────────────────────────────────

def test_reflect_returns_result(reflector, client):
    expected = ReflectionResult(
        root_cause="File missing",
        self_criticism="Bad assumption",
        new_strategy=["Try /tmp/correct.csv"],
        confidence=80,
    )
    client.messages.parse.return_value = _mock_parse_response(expected)

    result = reflector.reflect(
        tool_name="read_file",
        tool_input={"path": "/tmp/wrong.csv"},
        error_message="FileNotFoundError: No such file",
        task="read a file",
        attempt_number=1,
    )

    assert result.confidence == 80
    assert result.root_cause == "File missing"
    client.messages.parse.assert_called_once()


def test_reflect_attempt1_calls_haiku(reflector, client):
    result = ReflectionResult(
        root_cause="x", self_criticism="y", new_strategy=["z"], confidence=70
    )
    client.messages.parse.return_value = _mock_parse_response(result)

    reflector.reflect("tool", {}, "error", "task", attempt_number=1)

    call_kwargs = client.messages.parse.call_args
    assert call_kwargs.kwargs["model"] == Reflector.HAIKU


def test_reflect_attempt3_passes_thinking(reflector, client):
    result = ReflectionResult(
        root_cause="x", self_criticism="y", new_strategy=["z"], confidence=70
    )
    client.messages.parse.return_value = _mock_parse_response(result)

    reflector.reflect("tool", {}, "error", "task", attempt_number=3)

    call_kwargs = client.messages.parse.call_args
    assert call_kwargs.kwargs.get("thinking") == {"type": "adaptive"}


# ── LowConfidenceError ─────────────────────────────────────────────────────────

def test_reflect_raises_low_confidence(reflector, client):
    low = ReflectionResult(
        root_cause="x",
        self_criticism="y",
        new_strategy=["z"],
        confidence=CONFIDENCE_GATE - 1,
    )
    client.messages.parse.return_value = _mock_parse_response(low)

    with pytest.raises(LowConfidenceError) as exc_info:
        reflector.reflect("tool", {}, "error", "task", attempt_number=1)

    assert exc_info.value.reflection.confidence == CONFIDENCE_GATE - 1


def test_reflect_at_confidence_gate_does_not_raise(reflector, client):
    at_gate = ReflectionResult(
        root_cause="x",
        self_criticism="y",
        new_strategy=["z"],
        confidence=CONFIDENCE_GATE,
    )
    client.messages.parse.return_value = _mock_parse_response(at_gate)

    # Should NOT raise
    result = reflector.reflect("tool", {}, "error", "task", attempt_number=1)
    assert result.confidence == CONFIDENCE_GATE


def test_low_confidence_error_stores_reflection():
    r = ReflectionResult(
        root_cause="x", self_criticism="y", new_strategy=[], confidence=10
    )
    err = LowConfidenceError("low confidence", r)
    assert err.reflection is r


# ── reflect() — parse returns None ────────────────────────────────────────────

def test_reflect_raises_if_parsed_output_none(reflector, client):
    resp = MagicMock()
    resp.parsed_output = None
    client.messages.parse.return_value = resp

    with pytest.raises(ValueError, match="None"):
        reflector.reflect("tool", {}, "error", "task", attempt_number=1)


# ── format_for_injection ──────────────────────────────────────────────────────

def test_format_for_injection_contains_all_fields(reflector):
    r = ReflectionResult(
        root_cause="File not found",
        self_criticism="Wrong path assumed",
        new_strategy=["List directory", "Use absolute path"],
        confidence=80,
    )
    text = reflector.format_for_injection(r)
    assert "File not found" in text
    assert "Wrong path assumed" in text
    assert "List directory" in text
    assert "Use absolute path" in text
    assert "80" in text


def test_format_for_injection_numbered_steps(reflector):
    r = ReflectionResult(
        root_cause="x", self_criticism="y",
        new_strategy=["step A", "step B", "step C"],
        confidence=50,
    )
    text = reflector.format_for_injection(r)
    assert "1." in text
    assert "2." in text
    assert "3." in text
