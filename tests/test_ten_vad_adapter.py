import asyncio

import numpy as np
import pytest
from livekit import rtc
from livekit.agents import vad

import ten_vad_adapter


class _FakeTenVad:
    def __init__(self, hop_size: int, threshold: float) -> None:
        self.hop_size = hop_size
        self.threshold = threshold

    def process(self, audio_data: np.ndarray) -> tuple[float, int]:
        assert len(audio_data) == self.hop_size
        has_voice = int(np.max(np.abs(audio_data)) > 0)
        return (0.9 if has_voice else 0.1, has_voice)


def _frame(samples: np.ndarray, sample_rate: int = 16000) -> rtc.AudioFrame:
    pcm = samples.astype(np.int16, copy=False)
    return rtc.AudioFrame(
        data=pcm.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=len(pcm),
    )


@pytest.mark.asyncio
async def test_ten_vad_stream_emits_start_and_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ten_vad_adapter, "TenVad", _FakeTenVad)

    model = ten_vad_adapter.TenLiveKitVAD(
        sample_rate=16000,
        hop_size=256,
        threshold=0.5,
        min_speech_duration=0.016,
        min_silence_duration=0.032,
    )
    stream = model.stream()

    speech = np.ones(256, dtype=np.int16)
    silence = np.zeros(256, dtype=np.int16)
    stream.push_frame(_frame(speech))
    stream.push_frame(_frame(silence))
    stream.push_frame(_frame(silence))

    events: list[vad.VADEvent] = []
    for _ in range(8):
        event = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
        events.append(event)
        if event.type == vad.VADEventType.END_OF_SPEECH:
            break

    await stream.aclose()

    assert any(e.type == vad.VADEventType.INFERENCE_DONE for e in events)
    assert any(e.type == vad.VADEventType.START_OF_SPEECH for e in events)
    assert any(e.type == vad.VADEventType.END_OF_SPEECH for e in events)
