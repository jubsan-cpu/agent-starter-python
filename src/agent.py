import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
)
from livekit.plugins import cartesia, elevenlabs, openai

from gtcrn_audio_pipeline import AudioPreprocessor16k, GTCRNAudioInput, GTCRNModel
from smart_turn_adapter import SmartTurnConfig
from ten_vad_adapter import TenLiveKitVAD

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    """Prewarm TEN VAD and GTCRN model on process start."""
    # Smart Turn Configuration from ENV
    smart_turn_enabled = os.getenv("SMART_TURN_ENABLED", "true").lower() == "true"
    stop_secs = float(os.getenv("SMART_TURN_STOP_SECS", "0.8"))
    max_duration_secs = float(os.getenv("SMART_TURN_MAX_DURATION_SECS", "8.0"))
    model_path = os.getenv("SMART_TURN_MODEL_PATH")
    vad_min_speech_secs = float(os.getenv("VAD_MIN_SPEECH_SECS", "0.18"))

    st_config = SmartTurnConfig(
        enabled=smart_turn_enabled,
        stop_secs=stop_secs,
        max_duration_secs=max_duration_secs,
        model_path=model_path,
    )

    proc.userdata["vad"] = TenLiveKitVAD(
        smart_turn=st_config,
        min_speech_duration=vad_min_speech_secs,
    )
    proc.userdata["gtcrn_model"] = GTCRNModel()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        # STT — ElevenLabs Scribe v2 realtime
        stt=elevenlabs.STT(model_id="scribe_v2_realtime"),
        # LLM — OpenAI
        llm=openai.LLM(model="gpt-4o-mini"),
        # TTS — Cartesia Sonic 3
        tts=cartesia.TTS(model="sonic-3"),
        # TEN VAD — prewarmed, now receives enhanced audio (16k).
        # End-of-turn is gated by Smart Turn inside the adapter.
        vad=ctx.proc.userdata["vad"],
        # Allow LLM to generate while waiting for end of turn
        preemptive_generation=True,
    )

    # Start the session — RoomIO handles the raw audio input
    await session.start(
        agent=Assistant(),
        room=ctx.room,
    )

    # Wrap the RoomIO audio input with our GTCRN preprocessor.
    # This inserts the (soxr→)GTCRN chain BEFORE the VAD/STT fork:
    #   raw rtc (RoomIO) → GTCRNAudioInput (soxr?→GTCRN_16k) → VAD → STT
    room_audio = session.input.audio
    if room_audio is not None:
        # Per-session preprocessor (owns its own buffer, shares the GTCRN model)
        preprocessor = AudioPreprocessor16k(ctx.proc.userdata["gtcrn_model"])
        session.input.audio = GTCRNAudioInput(
            source=room_audio,
            preprocessor=preprocessor,
        )
        logger.info("GTCRN audio preprocessing inserted into pipeline")

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
