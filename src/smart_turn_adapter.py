import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger = logging.getLogger("smart-turn")


@dataclass
class SmartTurnConfig:
    enabled: bool = False
    prob_threshold: float = 0.6
    stop_secs: float = 1.7
    max_duration_secs: float = 8.0
    pre_speech_ms: float = 0.0
    model_path: str | None = None


class SmartTurnAnalyzer:
    """Async wrapper around Pipecat's LocalSmartTurnAnalyzerV3."""

    def __init__(self, config: SmartTurnConfig, sample_rate: int = 16000):
        self._config = config
        self._sample_rate = sample_rate
        self._analyzer = LocalSmartTurnAnalyzerV3(
            smart_turn_model_path=config.model_path,
            sample_rate=sample_rate,
        )

    @staticmethod
    def _prepare_audio(audio: np.ndarray) -> np.ndarray:
        """Normalize and coerce audio into 16k mono float32 in [-1, 1]."""
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if np.issubdtype(audio.dtype, np.integer):
            # int16 is expected in our pipeline.
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32, copy=False)
            max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
            if max_abs > 1.0:
                # Handle float arrays that still carry PCM-like ranges.
                audio = audio / 32768.0

        np.clip(audio, -1.0, 1.0, out=audio)
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _analyze_sync(self, audio: np.ndarray) -> float:
        """
        Run Smart Turn inference synchronously.

        Pipecat's API surface varies by version; support both:
        - public `analyze_audio(audio)` returning float or dict
        - private `_predict_endpoint(audio)` returning dict with probability
        """
        analyzer = self._analyzer
        result: Any

        if hasattr(analyzer, "analyze_audio"):
            result = analyzer.analyze_audio(audio)  # type: ignore[attr-defined]
            if isinstance(result, dict):
                return float(result.get("probability", 1.0))
            return float(result)

        if hasattr(analyzer, "_predict_endpoint"):
            result = analyzer._predict_endpoint(audio)  # type: ignore[attr-defined]
            if isinstance(result, dict):
                return float(result.get("probability", 1.0))
            return float(result)

        raise RuntimeError("Unsupported Smart Turn analyzer API: missing analyze method")

    async def analyze_async(self, audio: np.ndarray) -> float:
        """
        Analyze audio buffer for turn completion.
        Returns probability of turn completion (0.0 to 1.0).
        """
        if not self._config.enabled:
            # If disabled, treat as complete immediately (VAD-only behavior).
            return 1.0

        prepared = self._prepare_audio(audio)
        loop = asyncio.get_running_loop()
        prob = await loop.run_in_executor(None, self._analyze_sync, prepared)
        logger.debug(f"Smart Turn analysis: prob={prob:.3f}")
        return float(prob)
