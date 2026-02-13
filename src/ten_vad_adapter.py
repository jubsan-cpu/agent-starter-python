from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import numpy as np
from livekit import rtc
from livekit.agents import vad
from ten_vad import TenVad

from smart_turn_adapter import SmartTurnAnalyzer, SmartTurnConfig

logger = logging.getLogger("ten-vad")


@dataclass
class _TenVADOptions:
    sample_rate: int
    hop_size: int
    threshold: float
    min_speech_duration: float
    min_silence_duration: float
    smart_turn: SmartTurnConfig


class TenLiveKitVAD(vad.VAD):
    """TEN VAD adapter implementing LiveKit's VAD interface."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        hop_size: int = 256,
        threshold: float = 0.5,
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.55,
        smart_turn: SmartTurnConfig | None = None,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError("TEN VAD only supports 16kHz input")
        if hop_size <= 0:
            raise ValueError("hop_size must be > 0")
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if min_speech_duration < 0.0 or min_silence_duration < 0.0:
            raise ValueError("duration thresholds must be >= 0.0")

        super().__init__(
            capabilities=vad.VADCapabilities(update_interval=hop_size / sample_rate)
        )
        self._opts = _TenVADOptions(
            sample_rate=sample_rate,
            hop_size=hop_size,
            threshold=threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            smart_turn=smart_turn or SmartTurnConfig(),
        )

    @property
    def model(self) -> str:
        return "ten-vad"

    @property
    def provider(self) -> str:
        return "TEN Framework"

    def stream(self) -> vad.VADStream:
        return TenVADStream(self, self._opts)


class TenVADStream(vad.VADStream):
    def __init__(self, parent_vad: TenLiveKitVAD, opts: _TenVADOptions) -> None:
        super().__init__(parent_vad)
        self._opts = opts
        self._ten_vad = TenVad(opts.hop_size, opts.threshold)
        self._input_sample_rate = 0
        self._resampler: rtc.AudioResampler | None = None
        self._buffer = np.empty(0, dtype=np.int16)

        # Smart Turn additions
        self._smart_turn_analyzer = SmartTurnAnalyzer(opts.smart_turn, opts.sample_rate)
        self._inference_task: asyncio.Task | None = None

    @staticmethod
    def _frame_to_mono(frame: rtc.AudioFrame) -> np.ndarray:
        samples = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels <= 1:
            return samples
        return samples.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)

    @staticmethod
    def _to_audio_frame(samples: np.ndarray, sample_rate: int) -> rtc.AudioFrame:
        return rtc.AudioFrame(
            data=samples.astype(np.int16, copy=False).tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=len(samples),
        )

    def _push_model_input(self, frame: rtc.AudioFrame) -> None:
        mono = self._frame_to_mono(frame)
        mono_frame = self._to_audio_frame(mono, frame.sample_rate)

        if self._resampler is None:
            model_frames = [mono_frame]
        else:
            model_frames = self._resampler.push(mono_frame)

        for model_frame in model_frames:
            model_samples = np.frombuffer(model_frame.data, dtype=np.int16)
            if model_samples.size == 0:
                continue
            self._buffer = np.concatenate((self._buffer, model_samples))

    def _run_inference(self, chunk: np.ndarray) -> tuple[float, bool, float]:
        start = time.perf_counter()
        probability, flag = self._ten_vad.process(chunk)
        inference_duration = time.perf_counter() - start
        return probability, flag == 1, inference_duration

    def _cancel_inference(self) -> None:
        if self._inference_task is not None and not self._inference_task.done():
            self._inference_task.cancel()
        self._inference_task = None

    async def _main_task(self) -> None:
        speaking = False
        current_sample = 0
        timestamp = 0.0
        speech_duration = 0.0
        silence_duration = 0.0
        accumulated_speech = 0.0
        accumulated_silence = 0.0
        window_duration = self._opts.hop_size / self._opts.sample_rate

        async for input_frame in self._input_ch:
            if not isinstance(input_frame, rtc.AudioFrame):
                continue

            if not self._input_sample_rate:
                self._input_sample_rate = input_frame.sample_rate
                if self._input_sample_rate != self._opts.sample_rate:
                    self._resampler = rtc.AudioResampler(
                        input_rate=self._input_sample_rate,
                        output_rate=self._opts.sample_rate,
                        quality=rtc.AudioResamplerQuality.QUICK,
                    )
            elif input_frame.sample_rate != self._input_sample_rate:
                logger.error("a frame with another sample rate was already pushed")
                continue

            self._push_model_input(input_frame)

            while len(self._buffer) >= self._opts.hop_size:
                chunk = self._buffer[: self._opts.hop_size]
                self._buffer = self._buffer[self._opts.hop_size :]

                probability, is_speech, inference_duration = self._run_inference(chunk)
                current_sample += self._opts.hop_size
                timestamp += window_duration

                if speaking:
                    speech_duration += window_duration
                else:
                    silence_duration += window_duration

                chunk_frame = self._to_audio_frame(chunk, self._opts.sample_rate)
                self._event_ch.send_nowait(
                    vad.VADEvent(
                        type=vad.VADEventType.INFERENCE_DONE,
                        samples_index=current_sample,
                        timestamp=timestamp,
                        speech_duration=speech_duration,
                        silence_duration=silence_duration,
                        probability=probability,
                        inference_duration=inference_duration,
                        frames=[chunk_frame],
                        speaking=speaking,
                        raw_accumulated_speech=accumulated_speech,
                        raw_accumulated_silence=accumulated_silence,
                    )
                )

                immediate_complete = False
                if self._opts.smart_turn.enabled:
                    try:
                        immediate_complete = self._smart_turn_analyzer.append_audio(
                            chunk, is_speech
                        )
                    except Exception:
                        logger.exception("Error appending audio to Smart Turn")

                if is_speech:
                    if speaking and self._inference_task is not None:
                        self._cancel_inference()

                    accumulated_speech += window_duration
                    accumulated_silence = 0.0
                    if not speaking and (
                        accumulated_speech >= self._opts.min_speech_duration
                    ):
                        speaking = True
                        speech_duration = accumulated_speech
                        silence_duration = 0.0
                        self._event_ch.send_nowait(
                            vad.VADEvent(
                                type=vad.VADEventType.START_OF_SPEECH,
                                samples_index=current_sample,
                                timestamp=timestamp,
                                speech_duration=speech_duration,
                                silence_duration=silence_duration,
                                frames=[chunk_frame],
                                speaking=True,
                            )
                        )
                        if self._opts.smart_turn.enabled:
                            self._smart_turn_analyzer.update_vad_start_secs(
                                self._opts.min_speech_duration
                            )
                else:
                    accumulated_silence += window_duration
                    accumulated_speech = 0.0

                    if speaking and (
                        accumulated_silence >= self._opts.min_silence_duration
                    ):
                        if self._opts.smart_turn.enabled:
                            is_complete = immediate_complete
                            if (
                                not is_complete
                                and self._inference_task is None
                            ):
                                self._inference_task = asyncio.create_task(
                                    self._smart_turn_analyzer.analyze_end_of_turn()
                                )

                            if not is_complete:
                                if self._inference_task is None:
                                    continue
                                if not self._inference_task.done():
                                    continue
                                if self._inference_task.cancelled():
                                    self._inference_task = None
                                    continue
                                try:
                                    is_complete = bool(self._inference_task.result())
                                except Exception:
                                    logger.exception("Error during Smart Turn analysis")
                                    is_complete = True
                                finally:
                                    self._inference_task = None

                            if not is_complete:
                                continue

                        speaking = False
                        silence_duration = accumulated_silence
                        self._event_ch.send_nowait(
                            vad.VADEvent(
                                type=vad.VADEventType.END_OF_SPEECH,
                                samples_index=current_sample,
                                timestamp=timestamp,
                                speech_duration=max(
                                    0.0, speech_duration - accumulated_silence
                                ),
                                silence_duration=silence_duration,
                                frames=[chunk_frame],
                                speaking=False,
                            )
                        )
                        speech_duration = 0.0
                        accumulated_silence = 0.0
                        self._cancel_inference()
                        if self._opts.smart_turn.enabled:
                            self._smart_turn_analyzer.clear()

        self._cancel_inference()
