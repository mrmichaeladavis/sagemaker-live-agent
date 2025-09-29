import asyncio
import logging
from collections.abc import AsyncIterable

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("pre-reseponse-agent")

load_dotenv()


class PreResponseAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful assistant, respond in 10 words or less",
            llm=openai.LLM.with_ollama(
                model="llama3.2:1b", base_url="http://container02:11435/v1"
            ),
        )
        # We commented out the parallel LLM call to be able to run all the services locally
        # self._fast_llm = openai.LLM.with_ollama(model="llama3.2:1b", base_url="http://ollama:11435/v1")
        # self._fast_llm_prompt = llm.ChatMessage(
        #     role="system",
        #     content=[
        #         "Generate a very short instant response to the user's message with 5 to 10 words.",
        #         "Do not answer the questions directly. Examples: OK, Hm..., let me think about that, "
        #         "wait a moment, that's a good question, etc.",
        #     ],
        # )

    # async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage):
    #     # Create a short "silence filler" response to quickly acknowledge the user's input
    #     fast_llm_ctx = turn_ctx.copy(
    #         exclude_instructions=True, exclude_function_call=True
    #     ).truncate(max_items=3)
    #     fast_llm_ctx.items.insert(0, self._fast_llm_prompt)
    #     fast_llm_ctx.items.append(new_message)

    #     # # Intentionally not awaiting SpeechHandle to allow the main response generation to
    #     # # run concurrently
    #     # self.session.say(
    #     #     self._fast_llm.chat(chat_ctx=fast_llm_ctx).to_str_iterable(),
    #     #     add_to_chat_ctx=False,
    #     # )

    #     # Alternatively, if you want the reply to be aware of this "silence filler" response,
    #     # you can await the fast llm done and add the message to the turn context. But note
    #     # that not all llm supports completing from an existing assistant message.

    #     fast_llm_fut = asyncio.Future[str]()

    #     async def _fast_llm_reply() -> AsyncIterable[str]:
    #         filler_response: str = ""
    #         async for chunk in self._fast_llm.chat(chat_ctx=fast_llm_ctx).to_str_iterable():
    #             filler_response += chunk
    #             yield chunk
    #         fast_llm_fut.set_result(filler_response)

    #     self.session.say(_fast_llm_reply(), add_to_chat_ctx=False)
    #     filler_response = await fast_llm_fut

    #     logger.info(f"Fast response: {filler_response}")
    #     turn_ctx.add_message(role="assistant", content=filler_response, interrupted=False)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        # stt=deepgram.STT(),
        stt=openai.STT(base_url="http://container02:8000/v1", model="whisper-1"),
        tts=openai.TTS(
            base_url="http://container02:8880/v1", model="kokoro", voice="af_nova"
        ),  # lightweight open source tts
        vad=silero.VAD.load(),
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def handle_metrics(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    await session.start(PreResponseAgent(), room=ctx.room)

    ctx.add_shutdown_callback(log_usage)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
