"""
Streaming GTCRN audio preprocessing for voice AI.

Pipeline:
raw rtc -> (soxr to 16k if needed) -> GTCRN stream model -> VAD(16k) -> STT
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
import soxr
import torch
from livekit import rtc
from livekit.agents.voice.io import AudioInput

from vendor.gtcrn import GTCRN
from vendor.gtcrn_stream import StreamGTCRN
from vendor.stream_modules.convert import convert_to_stream

logger = logging.getLogger("gtcrn-pipeline")

GTCRN_SR = 16000
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512
_EPS = 1e-8
_DEFAULT_SHA256 = "a630d992cf792daf4ce2bb5bcf9c4d389f740a8f09c6e0971184697fe6371b79"


@dataclass
class GTCRNStreamState:
    conv_cache: torch.Tensor
    tra_cache: torch.Tensor
    inter_cache: torch.Tensor
    prev_samples: np.ndarray
    ola_signal: np.ndarray
    ola_norm: np.ndarray


class GTCRNModel:
    """Shared GTCRN model holder (loaded once in prewarm)."""

    def __init__(self) -> None:
        self._device = "cpu"
        checkpoint_path = self._resolve_checkpoint_path()
        self._validate_checkpoint(checkpoint_path)
        self._stream_model = self._load_stream_model(checkpoint_path)
        self._window = torch.hann_window(WIN_LENGTH).pow(0.5).to(self._device)
        self._window_np = self._window.detach().cpu().numpy().astype(np.float32)
        self._window_sq = self._window_np * self._window_np
        logger.info("GTCRN stream model ready")

    def _resolve_checkpoint_path(self) -> str:
        default_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "assets",
                "models",
                "gtcrn",
                "model_trained_on_dns3.tar",
            )
        )
        checkpoint_path = os.environ.get("GTCRN_CHECKPOINT_PATH", default_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"GTCRN checkpoint not found at: {checkpoint_path}")
        return checkpoint_path

    def _validate_checkpoint(self, checkpoint_path: str) -> None:
        expected = os.environ.get("GTCRN_CHECKPOINT_SHA256", _DEFAULT_SHA256)
        digest = hashlib.sha256()
        with open(checkpoint_path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(
                "GTCRN checkpoint checksum mismatch: "
                f"expected={expected} actual={actual} path={checkpoint_path}"
            )

    def _load_stream_model(self, checkpoint_path: str) -> StreamGTCRN:
        logger.info("Loading GTCRN checkpoint from %s", checkpoint_path)
        base_model = GTCRN().to(self._device).eval()
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        base_model.load_state_dict(checkpoint["model"])
        stream_model = StreamGTCRN().to(self._device).eval()
        convert_to_stream(stream_model, base_model)
        return stream_model

    def create_stream_state(self) -> GTCRNStreamState:
        return GTCRNStreamState(
            conv_cache=torch.zeros(2, 1, 16, 16, 33, device=self._device),
            tra_cache=torch.zeros(2, 3, 1, 1, 16, device=self._device),
            inter_cache=torch.zeros(2, 1, 33, 16, device=self._device),
            prev_samples=np.zeros(N_FFT - HOP_LENGTH, dtype=np.float32),
            ola_signal=np.zeros(N_FFT, dtype=np.float32),
            ola_norm=np.zeros(N_FFT, dtype=np.float32),
        )

    def enhance_hop_16k(
        self, state: GTCRNStreamState, hop_samples: np.ndarray
    ) -> np.ndarray:
        """Enhance one 16ms hop (256 samples at 16k) with streaming GTCRN."""
        if hop_samples.shape[0] != HOP_LENGTH:
            raise ValueError(
                f"Expected hop size {HOP_LENGTH}, got {hop_samples.shape[0]}"
            )

        frame = np.concatenate((state.prev_samples, hop_samples))
        state.prev_samples = hop_samples.copy()

        frame_tensor = torch.from_numpy(frame).to(self._device).unsqueeze(0)
        spec = torch.stft(
            frame_tensor,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self._window,
            center=False,
            return_complex=False,
        )

        with torch.no_grad():
            enhanced_spec, state.conv_cache, state.tra_cache, state.inter_cache = (
                self._stream_model(
                    spec,
                    state.conv_cache,
                    state.tra_cache,
                    state.inter_cache,
                )
            )

        enhanced_complex = torch.complex(
            enhanced_spec[0, :, 0, 0],
            enhanced_spec[0, :, 0, 1],
        )
        enhanced_frame = (
            torch.fft.irfft(enhanced_complex, n=N_FFT)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        enhanced_frame *= self._window_np

        state.ola_signal += enhanced_frame
        state.ola_norm += self._window_sq
        out_hop = state.ola_signal[:HOP_LENGTH] / np.maximum(
            state.ola_norm[:HOP_LENGTH], _EPS
        )

        state.ola_signal[:-HOP_LENGTH] = state.ola_signal[HOP_LENGTH:]
        state.ola_signal[-HOP_LENGTH:] = 0.0
        state.ola_norm[:-HOP_LENGTH] = state.ola_norm[HOP_LENGTH:]
        state.ola_norm[-HOP_LENGTH:] = 0.0
        return out_hop.astype(np.float32, copy=False)


class AudioPreprocessor16k:
    """Per-session streaming preprocessor."""

    def __init__(self, model: GTCRNModel) -> None:
        self._model = model
        self._stream_state = model.create_stream_state()
        self._hop_buffer = np.empty(0, dtype=np.float32)

    def process(
        self, pcm_data: bytes, sample_rate: int, num_channels: int
    ) -> list[tuple[np.ndarray, int]]:
        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        if num_channels > 1:
            audio = audio.reshape(-1, num_channels).mean(axis=1)
        if sample_rate != GTCRN_SR:
            audio = soxr.resample(audio, sample_rate, GTCRN_SR, quality=soxr.HQ)

        self._hop_buffer = np.concatenate((self._hop_buffer, audio))
        results: list[tuple[np.ndarray, int]] = []

        while self._hop_buffer.size >= HOP_LENGTH:
            hop = self._hop_buffer[:HOP_LENGTH]
            self._hop_buffer = self._hop_buffer[HOP_LENGTH:]
            enhanced = self._model.enhance_hop_16k(self._stream_state, hop)
            pcm_int16 = (enhanced * 32768.0).clip(-32768, 32767).astype(np.int16)
            results.append((pcm_int16, GTCRN_SR))

        return results


class GTCRNAudioInput(AudioInput):
    """LiveKit AudioInput wrapper that runs streaming GTCRN preprocessing."""

    def __init__(self, *, source: AudioInput, preprocessor: AudioPreprocessor16k) -> None:
        super().__init__(label="GTCRN", source=source)
        self._preprocessor = preprocessor
        self._out_queue: asyncio.Queue[rtc.AudioFrame] = asyncio.Queue()

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        while self._out_queue.empty():
            if self.source is None:
                raise StopAsyncIteration
            raw_frame = await self.source.__anext__()
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
