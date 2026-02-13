from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger = logging.getLogger("smart-turn")


@dataclass
class SmartTurnConfig:
    enabled: bool = False
    stop_secs: float = 0.8
    max_duration_secs: float = 8.0
    pre_speech_ms: float = 0.0
    model_path: str | None = None


class SmartTurnAnalyzer:
    """Wrapper around Pipecat BaseSmartTurn APIs."""

    def __init__(self, config: SmartTurnConfig, sample_rate: int = 16000):
        self._config = config
        self._sample_rate = sample_rate
        params = SmartTurnParams(
            stop_secs=config.stop_secs,
            pre_speech_ms=config.pre_speech_ms,
            max_duration_secs=config.max_duration_secs,
        )
        self._analyzer = LocalSmartTurnAnalyzerV3(
            smart_turn_model_path=config.model_path,
            sample_rate=sample_rate,
            params=params,
        )
        self._analyzer.set_sample_rate(sample_rate)

    @staticmethod
    def _to_pcm16_bytes(audio: np.ndarray) -> bytes:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if np.issubdtype(audio.dtype, np.integer):
            audio_i16 = audio.astype(np.int16, copy=False)
            return audio_i16.tobytes()
        audio_f32 = audio.astype(np.float32, copy=False)
        np.clip(audio_f32, -1.0, 1.0, out=audio_f32)
        audio_i16 = (audio_f32 * 32768.0).clip(-32768, 32767).astype(np.int16)
        return audio_i16.tobytes()

    def append_audio(self, chunk: np.ndarray, is_speech: bool) -> bool:
        if not self._config.enabled:
            return True
        state = self._analyzer.append_audio(self._to_pcm16_bytes(chunk), is_speech)
        return state == EndOfTurnState.COMPLETE

    async def analyze_end_of_turn(self) -> bool:
        if not self._config.enabled:
            return True
        state, _ = await self._analyzer.analyze_end_of_turn()
        return state == EndOfTurnState.COMPLETE

    def update_vad_start_secs(self, vad_start_secs: float) -> None:
        self._analyzer.update_vad_start_secs(vad_start_secs)

    def clear(self) -> None:
        self._analyzer.clear()
