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

from audio_pipeline import AudioPreprocessor, DFNAudioInput, DFNModel
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
    """Prewarm TEN VAD and DFN model on process start."""
    # Smart Turn Configuration from ENV
    smart_turn_enabled = os.getenv("SMART_TURN_ENABLED", "true").lower() == "true"
    prob_threshold = float(os.getenv("SMART_TURN_PROB_THRESHOLD", "0.6"))
    stop_secs = float(os.getenv("SMART_TURN_STOP_SECS", "1.7"))
    max_duration_secs = float(os.getenv("SMART_TURN_MAX_DURATION_SECS", "8.0"))
    model_path = os.getenv("SMART_TURN_MODEL_PATH")

    st_config = SmartTurnConfig(
        enabled=smart_turn_enabled,
        prob_threshold=prob_threshold,
        stop_secs=stop_secs,
        max_duration_secs=max_duration_secs,
        model_path=model_path,
    )

    proc.userdata["vad"] = TenLiveKitVAD(smart_turn=st_config)
    proc.userdata["dfn_model"] = DFNModel(model_name="DeepFilterNet2", atten_lim_db=90)


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
        llm=openai.LLM(model="gpt-5-nano"),
        # TTS — Cartesia Sonic 3
        tts=cartesia.TTS(model="sonic-3"),
        # TEN VAD — prewarmed, now receives denoised audio.
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

    # Wrap the RoomIO audio input with our DFN preprocessor.
    # This inserts the soxr→DFN→soxr chain BEFORE the VAD/STT fork:
    #   raw rtc (RoomIO) → DFNAudioInput (soxr→DFN→soxr) → VAD → STT
    room_audio = session.input.audio
    if room_audio is not None:
        # Per-session preprocessor (owns its own buffer, shares the DFN model)
        preprocessor = AudioPreprocessor(ctx.proc.userdata["dfn_model"])
        session.input.audio = DFNAudioInput(
            source=room_audio,
            preprocessor=preprocessor,
        )
        logger.info("DFN audio preprocessing inserted into pipeline")

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
