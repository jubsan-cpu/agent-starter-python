from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.voice.io import AudioInput

from gtcrn_audio_pipeline import (
    GTCRN_SR,
    HOP_LENGTH,
    AudioPreprocessor16k,
    GTCRNAudioInput,
)


@dataclass
class _FakeState:
    seen_hops: int = 0


class _FakeGTCRNModel:
    def create_stream_state(self) -> _FakeState:
        return _FakeState()

    def enhance_hop_16k(self, state: _FakeState, hop_samples: np.ndarray) -> np.ndarray:
        state.seen_hops += 1
        return hop_samples


@pytest.fixture
def preprocessor() -> AudioPreprocessor16k:
    return AudioPreprocessor16k(_FakeGTCRNModel())


def test_process_16k_no_resample(preprocessor: AudioPreprocessor16k) -> None:
    input_data = np.zeros(HOP_LENGTH * 10, dtype=np.int16).tobytes()
    chunks = preprocessor.process(input_data, 16000, 1)
    assert len(chunks) == 10
    for data, sr in chunks:
        assert sr == GTCRN_SR
        assert data.dtype == np.int16
        assert len(data) == HOP_LENGTH


def test_process_48k_resamples_to_16k(preprocessor: AudioPreprocessor16k) -> None:
    # 7680 @ 48k -> 2560 @ 16k -> 10 hops of 256.
    input_data = np.zeros(7680, dtype=np.int16).tobytes()
    chunks = preprocessor.process(input_data, 48000, 1)
    assert len(chunks) == 10
    for data, sr in chunks:
        assert sr == GTCRN_SR
        assert len(data) == HOP_LENGTH


def test_buffering_until_one_hop(preprocessor: AudioPreprocessor16k) -> None:
    half_hop = np.zeros(HOP_LENGTH // 2, dtype=np.int16).tobytes()
    assert preprocessor.process(half_hop, 16000, 1) == []
    chunks = preprocessor.process(half_hop, 16000, 1)
    assert len(chunks) == 1
    assert len(chunks[0][0]) == HOP_LENGTH


def test_model_error_raises(preprocessor: AudioPreprocessor16k) -> None:
    def _boom(state: _FakeState, hop_samples: np.ndarray) -> np.ndarray:
        _ = (state, hop_samples)
        raise RuntimeError("forced model failure")

    preprocessor._model.enhance_hop_16k = _boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="forced model failure"):
        preprocessor.process(np.zeros(HOP_LENGTH, dtype=np.int16).tobytes(), 16000, 1)


@pytest.mark.asyncio
async def test_audio_input_adapter_outputs_16k_hops(
    preprocessor: AudioPreprocessor16k,
) -> None:
    mock_source = MagicMock(spec=AudioInput)
    frames = [
        rtc.AudioFrame(
            data=np.zeros(480, dtype=np.int16).tobytes(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=480,
        )
        for _ in range(4)
    ]
    frame_iter = iter(frames)

    async def _next_frame() -> rtc.AudioFrame:
        try:
            return next(frame_iter)
        except StopIteration as err:
            raise StopAsyncIteration from err

    mock_source.__anext__.side_effect = _next_frame
    adapter = GTCRNAudioInput(source=mock_source, preprocessor=preprocessor)

    out = []
    async for frame in adapter:
        out.append(frame)
        if len(out) >= 1:
            break

    assert len(out) == 1
    assert out[0].sample_rate == GTCRN_SR
    assert out[0].samples_per_channel == HOP_LENGTH
