import asyncio
from typing import ClassVar

import numpy as np
import pytest
from livekit import rtc
from livekit.agents import vad

import ten_vad_adapter
from smart_turn_adapter import SmartTurnConfig


class _FakeTenVad:
    def __init__(self, hop_size: int, threshold: float) -> None:
        self.hop_size = hop_size
        self.threshold = threshold

    def process(self, audio_data: np.ndarray) -> tuple[float, int]:
        assert len(audio_data) == self.hop_size
        has_voice = int(np.max(np.abs(audio_data)) > 0)
        return (0.9 if has_voice else 0.1, has_voice)


class _FakeSmartTurnAnalyzer:
    outcomes: ClassVar[list[bool]] = [True]
    delay_s: ClassVar[float] = 0.0
    calls: ClassVar[int] = 0
    append_calls: ClassVar[int] = 0

    def __init__(self, config: SmartTurnConfig, sample_rate: int = 16000) -> None:
        self._config = config
        self._sample_rate = sample_rate

    def append_audio(self, chunk: np.ndarray, is_speech: bool) -> bool:
        type(self).append_calls += 1
        _ = (chunk, is_speech)
        return False

    async def analyze_end_of_turn(self) -> bool:
        type(self).calls += 1
        await asyncio.sleep(type(self).delay_s)
        if type(self).outcomes:
            return type(self).outcomes.pop(0)
        return True

    def clear(self) -> None:
        return

    def update_vad_start_secs(self, vad_start_secs: float) -> None:
        _ = vad_start_secs
        return


class _FakeSmartTurnAnalyzerTimeout(_FakeSmartTurnAnalyzer):
    silence_hops_required: ClassVar[int] = 3
    _silence_hops: int

    def __init__(self, config: SmartTurnConfig, sample_rate: int = 16000) -> None:
        super().__init__(config, sample_rate)
        self._silence_hops = 0

    def append_audio(self, chunk: np.ndarray, is_speech: bool) -> bool:
        type(self).append_calls += 1
        _ = chunk
        if is_speech:
            self._silence_hops = 0
            return False
        self._silence_hops += 1
        return self._silence_hops >= type(self).silence_hops_required


def _frame(samples: np.ndarray, sample_rate: int = 16000) -> rtc.AudioFrame:
    pcm = samples.astype(np.int16, copy=False)
    return rtc.AudioFrame(
        data=pcm.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=len(pcm),
    )


async def _collect_events(
    stream: ten_vad_adapter.TenVADStream,
    *,
    max_events: int = 40,
    timeout: float = 0.5,
    stop_on_end: bool = True,
) -> list[vad.VADEvent]:
    events: list[vad.VADEvent] = []
    for _ in range(max_events):
        try:
            event = await asyncio.wait_for(stream.__anext__(), timeout=timeout)
        except (asyncio.TimeoutError, StopAsyncIteration, asyncio.InvalidStateError):
            break
        events.append(event)
        if stop_on_end and event.type == vad.VADEventType.END_OF_SPEECH:
            break
    return events


def _reset_fake_smart_turn(*, outcomes: list[bool], delay_s: float = 0.0) -> None:
    _FakeSmartTurnAnalyzer.outcomes = outcomes
    _FakeSmartTurnAnalyzer.delay_s = delay_s
    _FakeSmartTurnAnalyzer.calls = 0
    _FakeSmartTurnAnalyzer.append_calls = 0


@pytest.mark.asyncio
async def test_smart_turn_complete_emits_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ten_vad_adapter, "TenVad", _FakeTenVad)
    monkeypatch.setattr(ten_vad_adapter, "SmartTurnAnalyzer", _FakeSmartTurnAnalyzer)
    _reset_fake_smart_turn(outcomes=[True], delay_s=0.001)

    model = ten_vad_adapter.TenLiveKitVAD(
        sample_rate=16000,
        hop_size=256,
        threshold=0.5,
        min_speech_duration=0.016,
        min_silence_duration=0.016,
        smart_turn=SmartTurnConfig(enabled=True, stop_secs=0.8),
    )
    stream = model.stream()

    speech = np.ones(256, dtype=np.int16)
    silence = np.zeros(256, dtype=np.int16)

    # Push speech chunk: expect inference + START_OF_SPEECH.
    stream.push_frame(_frame(speech))
    e1 = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
    e2 = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
    assert {e1.type, e2.type} == {
        vad.VADEventType.INFERENCE_DONE,
        vad.VADEventType.START_OF_SPEECH,
    }

    # Push first silence chunk to trigger Smart Turn inference.
    stream.push_frame(_frame(silence))
    _ = await asyncio.wait_for(stream.__anext__(), timeout=1.0)  # INFERENCE_DONE

    # Let the async Smart Turn task complete.
    await asyncio.sleep(0.01)

    # Next silence chunk should emit END_OF_SPEECH after gating.
    stream.push_frame(_frame(silence))
    events: list[vad.VADEvent] = []
    for _ in range(3):
        events.append(await asyncio.wait_for(stream.__anext__(), timeout=1.0))
        if events[-1].type == vad.VADEventType.END_OF_SPEECH:
            break
    await stream.aclose()

    assert any(e.type == vad.VADEventType.END_OF_SPEECH for e in events), (
        f"calls={_FakeSmartTurnAnalyzer.calls}, "
        f"events={[e.type.name for e in events]}"
    )
    assert _FakeSmartTurnAnalyzer.calls >= 1
    assert _FakeSmartTurnAnalyzer.append_calls >= 1


@pytest.mark.asyncio
async def test_smart_turn_base_timeout_emits_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ten_vad_adapter, "TenVad", _FakeTenVad)
    monkeypatch.setattr(
        ten_vad_adapter, "SmartTurnAnalyzer", _FakeSmartTurnAnalyzerTimeout
    )

    model = ten_vad_adapter.TenLiveKitVAD(
        sample_rate=16000,
        hop_size=256,
        threshold=0.5,
        min_speech_duration=0.016,
        min_silence_duration=0.016,
        smart_turn=SmartTurnConfig(enabled=True, stop_secs=0.8),
    )
    stream = model.stream()

    speech = np.ones(256, dtype=np.int16)
    silence = np.zeros(256, dtype=np.int16)
    stream.push_frame(_frame(speech))
    for _ in range(12):
        stream.push_frame(_frame(silence))

    events = await _collect_events(stream)
    await stream.aclose()

    end_event = next((e for e in events if e.type == vad.VADEventType.END_OF_SPEECH), None)
    assert end_event is not None


@pytest.mark.asyncio
async def test_resumed_speech_cancels_stale_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ten_vad_adapter, "TenVad", _FakeTenVad)
    monkeypatch.setattr(ten_vad_adapter, "SmartTurnAnalyzer", _FakeSmartTurnAnalyzer)
    _reset_fake_smart_turn(outcomes=[True, False], delay_s=0.08)

    model = ten_vad_adapter.TenLiveKitVAD(
        sample_rate=16000,
        hop_size=256,
        threshold=0.5,
        min_speech_duration=0.016,
        min_silence_duration=0.016,
        smart_turn=SmartTurnConfig(enabled=True, stop_secs=0.8),
    )
    stream = model.stream()

    speech = np.ones(256, dtype=np.int16)
    silence = np.zeros(256, dtype=np.int16)

    # First pause starts inference; resumed speech should cancel stale result.
    stream.push_frame(_frame(speech))
    stream.push_frame(_frame(silence))
    stream.push_frame(_frame(speech))
    for _ in range(4):
        stream.push_frame(_frame(silence))

    events = await _collect_events(stream, stop_on_end=False)
    # Allow any in-flight analysis coroutine to be scheduled.
    await asyncio.sleep(0)
    await stream.aclose()

    assert not any(e.type == vad.VADEventType.END_OF_SPEECH for e in events)
