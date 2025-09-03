#!/usr/bin/env python3

import logging
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict

from dotenv import load_dotenv
import os

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    RunContext,
    cli,
)
from livekit.agents.llm import function_tool

from livekit.plugins import openai, silero, deepgram
from livekit.plugins import noise_cancellation
from livekit.plugins import elevenlabs

load_dotenv()
logger = logging.getLogger(__name__)


# ---------- Instructions builder ----------

def build_instructions() -> str:
    with open("prompt.txt", "r", encoding="utf-8") as f:
        return f.read()

# ---------- Agent subclass ----------

class InterviewAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=build_instructions(),
        )

    async def on_enter(self) -> None:
        # Kick off the conversation immediately.
        await self.session.generate_reply(instructions="Behave as per the instructions given. Be friendly and engaging.")

# ---------- Worker entrypoint ----------

async def entrypoint(ctx: JobContext):
    logger.info(f"🎯 [WORKER] Starting interview worker for room: {ctx.room.name}")
    agent = InterviewAgent()

    session = AgentSession(
        turn_detection="stt",
        preemptive_generation=True,
        vad=silero.VAD.load(
            min_silence_duration=200,
            min_speech_duration=100,
        ),
        ## STT: Whisper via OpenAI
        stt=openai.STT(
            model="whisper-1",
            language="as"  # Assamese language code
        ),
        llm=openai.LLM(
            model="gpt-4o",
            temperature=0.0,
        ), 
        ## TTS: ElevenLabs with Assamese support
        tts=elevenlabs.TTS(
            model="eleven_flash_v2_5",
            voice_id="FGY2WhTYpPnrIDTdsKH5",
        ),
    )

    logger.info(f"🚀 [WORKER] Starting session for room: {ctx.room.name}")
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )
    # Connect the worker to the room after starting the session
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))