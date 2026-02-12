"""
Audio preprocessing pipeline for voice AI:
Raw RTC audio → soxr (→48kHz) → DeepFilterNet (denoise @ 48kHz) → soxr (→16kHz)

This module provides two components:
1. AudioPreprocessor — stateful soxr/DFN processing engine
2. DFNAudioInput   — an AudioInput wrapper that slots into the LiveKit
                     pipeline *before* the VAD/STT fork, so both get
                     denoised audio.

Pipeline order:  raw rtc → soxr → DFN → soxr → VAD → STT
"""

import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np
import soxr
import torch
from df.enhance import enhance, init_df
from livekit import rtc
from livekit.agents.voice.io import AudioInput

logger = logging.getLogger("audio-pipeline")

DEEPFILTER_SR = 48000
OUTPUT_SR = 16000

# 200ms chunks — balances latency vs DeepFilterNet quality
CHUNK_DURATION_S = 0.2
CHUNK_SAMPLES = int(DEEPFILTER_SR * CHUNK_DURATION_S)  # 9600 samples


class DFNModel:
    """
    Holds the shared DeepFilterNet model. Loaded once in prewarm(),
    safe to share across sessions (model inference is stateless).

    Args:
        atten_lim_db: Noise attenuation limit in dB.
            None  = unlimited suppression (most aggressive)
            6     = suppress at most 6 dB  (gentle, natural)
            12    = suppress at most 12 dB (moderate)
            20    = suppress at most 20 dB (strong)
    """

    def __init__(self, model_name: str = "DeepFilterNet3", atten_lim_db: float | None = None):
        self.model, self.df_state, _ = init_df(model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self._device)
        self.model.eval()
        self.atten_lim_db = atten_lim_db
        logger.info(
            f"DFNModel ready (device={self._device}, "
            f"atten_lim_db={atten_lim_db}, "
            f"chunk={CHUNK_DURATION_S * 1000:.0f}ms)"
        )


class AudioPreprocessor:
    """
    Per-session processor that chains soxr resampling and DeepFilterNet
    noise suppression. Each session must have its own instance because
    the internal buffer is stateful.
    """

    def __init__(self, dfn: DFNModel):
        self._dfn = dfn
        self._buffer = np.array([], dtype=np.float32)

    def process(
        self, pcm_data: bytes, sample_rate: int, num_channels: int
    ) -> list[tuple[np.ndarray, int]]:
        """
        Feed raw PCM int16 audio bytes. Returns list of
        (processed_pcm_int16_array, output_sample_rate) tuples.
        May return empty list if still buffering.
        """
        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        if num_channels > 1:
            audio = audio.reshape(-1, num_channels).mean(axis=1)

        # Step 1: soxr → 48kHz for DeepFilterNet
        if sample_rate != DEEPFILTER_SR:
            audio = soxr.resample(audio, sample_rate, DEEPFILTER_SR, quality=soxr.HQ)

        self._buffer = np.concatenate([self._buffer, audio])

        results = []
        while len(self._buffer) >= CHUNK_SAMPLES:
            chunk = self._buffer[:CHUNK_SAMPLES]
            self._buffer = self._buffer[CHUNK_SAMPLES:]

            try:
                processed = self._denoise_and_resample(chunk)
                results.append((processed, OUTPUT_SR))
            except Exception:
                logger.warning(
                    "DeepFilterNet failed on chunk, passing raw resampled audio"
                )
                fallback = soxr.resample(
                    chunk, DEEPFILTER_SR, OUTPUT_SR, quality=soxr.HQ
                )
                fallback_int16 = (
                    (fallback * 32768.0).clip(-32768, 32767).astype(np.int16)
                )
                results.append((fallback_int16, OUTPUT_SR))

        return results

    def _denoise_and_resample(self, chunk_48k: np.ndarray) -> np.ndarray:
        """DeepFilterNet denoise at 48kHz, then soxr downsample to 16kHz."""
        dfn = self._dfn
        with torch.no_grad():
            tensor = torch.from_numpy(chunk_48k).unsqueeze(0).to(dfn._device)
            enhanced = enhance(
                dfn.model, dfn.df_state, tensor, atten_lim_db=dfn.atten_lim_db
            )
            enhanced_np = enhanced.squeeze(0).cpu().numpy()

        # Step 3: soxr → 16kHz for VAD / STT
        enhanced_16k = soxr.resample(
            enhanced_np, DEEPFILTER_SR, OUTPUT_SR, quality=soxr.HQ
        )
        return (enhanced_16k * 32768.0).clip(-32768, 32767).astype(np.int16)


class DFNAudioInput(AudioInput):
    """
    LiveKit AudioInput that wraps a source AudioInput and applies the
    soxr → DeepFilterNet → soxr preprocessing chain to every frame.

    By sitting at the AudioInput level (before the VAD/STT fork in
    AgentSession._forward_audio_task), both VAD and STT receive
    denoised audio.

    Pipeline:  RoomIO → DFNAudioInput → push_audio() → VAD + STT
    """

    def __init__(self, *, source: AudioInput, preprocessor: AudioPreprocessor) -> None:
        super().__init__(label="DFN", source=source)
        self._preprocessor = preprocessor
        # Queue for processed frames (since one input frame may yield
        # zero or multiple output frames due to DFN buffering)
        self._out_queue: asyncio.Queue[rtc.AudioFrame] = asyncio.Queue()

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        # Drain any already-processed frames first
        while self._out_queue.empty():
            # Get next raw frame from the upstream source
            if self.source is None:
                raise StopAsyncIteration

            raw_frame = await self.source.__anext__()

            # Run through soxr → DFN → soxr
            chunks = self._preprocessor.process(
                bytes(raw_frame.data),
                raw_frame.sample_rate,
                raw_frame.num_channels,
            )

            for pcm_int16, sr in chunks:
                self._out_queue.put_nowait(
                    rtc.AudioFrame(
                        data=pcm_int16.tobytes(),
                        sample_rate=sr,
                        num_channels=1,
                        samples_per_channel=len(pcm_int16),
                    )
                )

        return self._out_queue.get_nowait()
