import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
)
from livekit.plugins import cartesia, elevenlabs, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from audio_pipeline import AudioPreprocessor, DFNAudioInput, DFNModel

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
    """Prewarm Silero VAD and DFN model on process start."""
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["dfn_model"] = DFNModel(atten_lim_db=20)


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
        # Turn detection
        turn_detection=MultilingualModel(),
        # Silero VAD — prewarmed, now receives denoised audio
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
