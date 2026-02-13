from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from livekit.agents import JobProcess


def test_prewarm_wiring() -> None:
    from agent import prewarm

    proc = MagicMock(spec=JobProcess)
    proc.userdata = {}

    with patch("agent.TenLiveKitVAD") as mock_vad, patch(
        "agent.GTCRNModel"
    ) as mock_gtcrn:
        prewarm(proc)

    assert "vad" in proc.userdata
    assert "gtcrn_model" in proc.userdata
    mock_vad.assert_called_once()
    mock_gtcrn.assert_called_once()


def test_prewarm_default_smart_turn_stop_secs(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent import prewarm

    monkeypatch.delenv("SMART_TURN_STOP_SECS", raising=False)

    proc = MagicMock(spec=JobProcess)
    proc.userdata = {}

    with patch("agent.TenLiveKitVAD") as mock_vad, patch("agent.GTCRNModel"):
        prewarm(proc)

    assert mock_vad.call_count == 1
    kwargs = mock_vad.call_args.kwargs
    assert kwargs["smart_turn"].stop_secs == pytest.approx(0.8)


def test_imports() -> None:
    from agent import AudioPreprocessor16k, GTCRNAudioInput

    assert GTCRNAudioInput is not None
    assert AudioPreprocessor16k is not None
