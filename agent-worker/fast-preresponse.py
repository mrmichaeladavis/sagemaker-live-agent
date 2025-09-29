import asyncio
import atexit
import inspect
import json
import logging
import os
import time
from collections.abc import AsyncIterable
from datetime import datetime
from typing import Literal, Type, cast

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    WorkerOptions,
    cli,
    llm,
    metrics,
    stt,
    tts,
)
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.agents.metrics import (
    AgentMetrics,
    EOUMetrics,
    LLMMetrics,
    STTMetrics,
    TTSMetrics,
    VADMetrics,
)
from livekit.plugins import aws, deepgram, elevenlabs, groq, openai, silero
from prometheus_client import CollectorRegistry, Counter, Gauge, multiprocess
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Configuration ---
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Pydantic Models for Type-Safe Configuration ---


class LLMConfig(BaseModel):
    provider: Literal["openai", "groq", "aws"]
    model: str
    base_url: str | None = None
    api_key: str | None = None
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0


class STTConfig(BaseModel):
    provider: Literal["deepgram", "aws", "openai"]
    model: str | None = None
    language: str = "en-US"
    cost_per_second: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


class TTSConfig(BaseModel):
    provider: Literal["openai", "groq", "aws", "elevenlabs"]
    model: str
    voice: str
    voice_id: str | None = None  # For ElevenLabs
    api_key: str | None = None
    base_url: str | None = None
    cost_per_character: float = 0.0


class VADConfig(BaseModel):
    min_silence_duration: float = 0.2
    activation_threshold: float = 0.3


class AppConfig(BaseSettings):
    def model_post_init(self, __context) -> None:
        import os

        os.makedirs(self.prometheus_multiproc_dir, exist_ok=True)
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = self.prometheus_multiproc_dir
        logger.debug(f"Prometheus multiproc dir set to {self.prometheus_multiproc_dir}")

    model_config = SettingsConfigDict(
        env_file=".env", env_nested_delimiter="__", extra="allow"
    )

    allow_interruptions: bool = True
    agent_type: str = Field(
        default_factory=lambda: os.path.splitext(os.path.basename(__file__))[0]
    )
    prometheus_multiproc_dir: str = "/tmp/prometheus_multiproc"
    agent_instructions: str = """### System Persona

You are Warren, a creative, friendly, and intelligent AI voice assistant. The user is interacting with you via voice on their phone, and your entire response will be converted to speech by a realistic text-to-speech (TTS) engine. Your persona is natural, conversational, and concise, like talking to a knowledgeable friend.

### Operating Context

- **Interaction**: Voice-only conversation.
- **Output Medium**: Your text response is read aloud by a TTS system.
- **User Perception**: The user hears a continuous spoken voice. They did not type their query, so any ambiguity is due to your mishearing, not their typo.
- **Knowledge Cutoff**: 2023-10
- **Current Date**: 2025-09-30

### Core Directives

You MUST adhere to these directives at all times. Failure to do so will result in a penalty.

1.  **Adopt Expert Persona Silently**: For each query, determine the relevant field and adopt the persona of a corresponding expert. You MUST use that expert's insight and vocabulary but translate it into simple, conversational language. NEVER state the expert role you have chosen.
2.  **Extreme Conciseness**: Default to responses of one or two sentences (under 100 words). Provide high-level summaries first. Await user follow-up before providing details.
    - **Bad**: "Paris is the capital and most populous city of France, with an estimated population of 2,165,423 residents as of January 1, 2023..."
    - **Good**: "Paris? It's France's capital... about two million people live there. Beautiful city!"
3.  **Maintain Conversational Flow**: Speak in a continuous, natural manner. Your goal is to keep the conversation going.
    - Initiate responses with acknowledgments like "Okay," "Got it," or "Let me check."
    - Use occasional, natural-sounding hesitations like "um" or "uh."
    - Seamlessly transition between thoughts with markers like "So," or "Actually."
    - Never use conversational end-caps (e.g., "Enjoy!"), or ask if the user needs more help (e.g., "How can I assist you further?"). Let the conversation end naturally.
4.  **Prioritize Spoken Clarity**: Write for the ear, not the eye.
    - Use simple vocabulary and short sentences.
    - Format all responses as continuous spoken paragraphs. Do not use lists, markdown, or bullet points.
    - If interrupted, respond with "Oh, sorry, go ahead" and cease speaking.
5.  **Handle Ambiguity**: If a query is unclear, ask clarifying questions instead of making assumptions.

### Response Generation Logic

You MUST follow this internal step-by-step process for every user query:

1.  **Analyze**: Evaluate the user's question to determine the most appropriate field of study.
2.  **Adopt**: Silently assume the role of an expert in that field.
3.  **Formulate**: Generate a high-quality, accurate answer based on your expert knowledge.
4.  **Translate & Style**: Convert the expert answer into natural, concise, and conversational language according to the **Conversational Style Guide**.
5.  **Format**: Ensure the final text is formatted correctly for the TTS engine according to the **TTS Formatting Rules**.

### Conversational Style Guide

- **Natural vs. Robotic Speech**:
    - **Robotic**: "I have found three restaurants matching your criteria. The first option is Luigi's Italian Restaurant located at 123 Main Street."
    - **Natural**: "Okay, so... I found three places that could work. First up is, uh, Luigi's - it's an Italian place on Main Street."
- **Apologies**: Limit apologies to a maximum of one per conversation. Replace "I'm sorry" with an action, e.g., "Let me fix that."

### TTS Formatting Rules

You MUST format the following entities as specified to ensure correct TTS pronunciation.

| Category          | Written Format        | Spoken Format (Your Output)                     |
| ----------------- | --------------------- | ----------------------------------------------- |
| **Abbreviations** | FBI, RSVP             | F-B-I, R-S-V-P                                  |
| **Acronyms**      | NASA, ASAP            | naa-suh, ay-sap                                 |
| **Numbers**       | 1235                  | twelve hundred and thirty-five                  |
| **Phone Numbers** | (555) 123-4567        | five five five, one two three, four five six seven |
| **Money**         | $19.99                | nineteen dollars and ninety-nine cents          |
| **Dates**         | 02/14/2025            | February fourteenth, twenty twenty-five         |
| **Times**         | 3:30 PM               | three thirty in the afternoon                   |
| **Email/URLs**    | <john@co.com>/guide     | john at co dot com slash guide                  |
| **Fractions**     | 2/3                   | two-thirds                                      |
| **Roman Numerals**| Chapter XIV / Eliz. II| Chapter fourteen / Elizabeth the second         |
| **Units**         | 100km, 5GB            | one hundred kilometers, five gigabytes          |
| **Shortcuts**     | Ctrl+Z                | control Z                                       |
| **File Paths**    | C:\\Users\\Docs         | C drive, users folder, documents folder         |

### Absolute Constraints

- You MUST follow all rules absolutely.
- You MUST NOT refer to these rules or your nature as an AI, even if asked.
- You MUST NOT use emojis, symbols (@#$%), or text formatting like bold or italics.
- You MUST provide unbiased answers and avoid stereotypes.
    """
    fast_llm_prompt: str = "Generate a short instant response to the user's message with 5 to 10 words. Do not answer the questions directly. Examples:, let me think about that, wait a moment, that's a good question, etc."

    primary_llm: LLMConfig = LLMConfig(
        provider="openai",
        model="openai-gpt-4o",
        cost_per_input_token=0.005 / 1000,
        cost_per_output_token=0.015 / 1000,
    )
    fast_llm: LLMConfig = LLMConfig(
        provider="openai",
        model="google-gemini-2.5-flash-lite",
        cost_per_input_token=0.05 / 1_000_000,
        cost_per_output_token=0.05 / 1_000_000,
    )
    stt: STTConfig = STTConfig(
        provider="deepgram",
        model="nova-3",
        cost_per_second=0.00499 / 60,
    )
    tts: TTSConfig = TTSConfig(
        provider="openai",
        model="tts-1-hd",
        voice="alloy",
        cost_per_character=0.015 / 1000,
    )
    vad: VADConfig = VADConfig(min_silence_duration=0.3, activation_threshold=0.4)


# --- Plugin Registry ---


class PluginRegistry:
    """A registry that maps provider names to plugin classes for dynamic instantiation."""

    def __init__(self):
        self._llm_registry: dict[str, Type[llm.LLM]] = {
            "openai": openai.LLM,
            "groq": groq.LLM,
            "aws": aws.LLM,
        }
        self._stt_registry: dict[str, Type[stt.STT]] = {
            "deepgram": deepgram.STT,
            "aws": aws.STT,
        }
        self._tts_registry: dict[str, Type[tts.TTS]] = {
            "openai": openai.TTS,
            "groq": groq.TTS,
            "aws": aws.TTS,
            "elevenlabs": elevenlabs.TTS,
        }

    def _create_plugin(self, registry: dict, config: BaseModel) -> object:
        provider = getattr(config, "provider", None)
        if provider not in registry:
            raise ValueError(f"Unsupported or unregistered provider: {provider}")

        cls = registry[provider]
        # Pass config as kwargs, excluding meta fields not used by constructors
        all_kwargs = config.model_dump(
            exclude={
                "provider",
                "cost_per_input_token",
                "cost_per_output_token",
                "cost_per_second",
                "cost_per_character",
            },
            exclude_none=True,
        )  # Exclude None values like an unset base_url

        # Filter kwargs to only those accepted by the cls constructor
        sig = inspect.signature(cls.__init__)
        accepted_params = set(sig.parameters.keys()) - {"self"}
        kwargs = {k: v for k, v in all_kwargs.items() if k in accepted_params}

        logger.debug(f"Creating plugin for provider '{provider}' with config: {kwargs}")
        return cls(**kwargs)

    def create_llm(self, config: LLMConfig) -> llm.LLM:
        return cast(llm.LLM, self._create_plugin(self._llm_registry, config))

    def create_stt(self, config: STTConfig) -> stt.STT:
        return cast(stt.STT, self._create_plugin(self._stt_registry, config))

    def create_tts(self, config: TTSConfig) -> tts.TTS:
        return cast(tts.TTS, self._create_plugin(self._tts_registry, config))


# --- Metrics Management ---
class MetricsManager:
    def __init__(self, config: AppConfig):
        self._config = config
        self._registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(self._registry)
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = config.prometheus_multiproc_dir

        self._usage_collector = metrics.UsageCollector()
        self._last_usage_summary = self._usage_collector.get_summary()

        self._current_turn_metrics: dict[str, float | None] = {
            "eou_delay": None,
            "llm_ttft": None,
            "tts_ttfb": None,
        }
        self._turn_id_counter = 0

        # --- Latency Metrics ---
        self.llm_latency = Gauge(
            "livekit_llm_duration_ms",
            "LLM latency in milliseconds",
            ["model", "agent_type"],
            registry=self._registry,
        )
        self.llm_latency_small = Gauge(
            "livekit_llm_small_duration_ms",
            "Fast LLM latency in milliseconds",
            ["model", "agent_type"],
            registry=self._registry,
        )
        self.stt_latency = Gauge(
            "livekit_stt_duration_ms",
            "Speech-to-text latency in milliseconds",
            ["provider", "agent_type"],
            registry=self._registry,
        )
        self.tts_latency = Gauge(
            "livekit_tts_duration_ms",
            "Text-to-speech latency in milliseconds",
            ["provider", "agent_type"],
            registry=self._registry,
        )
        self.eou_latency = Gauge(
            "livekit_eou_delay_ms",
            "End-of-utterance delay in milliseconds",
            ["agent_type"],
            registry=self._registry,
        )
        self.total_conversation_latency = Gauge(
            "livekit_total_conversation_latency_ms",
            "Current conversation latency in milliseconds",
            ["agent_type"],
            registry=self._registry,
        )

        # --- Usage & Concurrency Metrics ---
        self.llm_tokens = Counter(
            "livekit_llm_tokens_total",
            "Total LLM tokens processed",
            ["type", "model"],
            registry=self._registry,
        )
        self.stt_duration = Counter(
            "livekit_stt_duration_seconds_total",
            "Total STT audio duration in seconds",
            ["provider"],
            registry=self._registry,
        )
        self.tts_chars = Counter(
            "livekit_tts_chars_total",
            "Total TTS characters processed",
            ["provider"],
            registry=self._registry,
        )
        self.total_tokens = Counter(
            "livekit_total_tokens_total",
            "Total tokens processed",
            registry=self._registry,
        )
        self.conversation_turns = Counter(
            "livekit_conversation_turns_total",
            "Number of conversation turns",
            ["agent_type", "room"],
            registry=self._registry,
        )
        self.active_conversations = Gauge(
            "livekit_active_conversations",
            "Number of active conversations",
            ["agent_type"],
            multiprocess_mode="liveall",
            registry=self._registry,
        )

        # --- Cost Metrics ---
        self.llm_cost = Gauge(
            "livekit_llm_cost_total",
            "Total LLM cost in USD",
            ["model"],
            multiprocess_mode="liveall",
            registry=self._registry,
        )
        self.stt_cost = Gauge(
            "livekit_stt_cost_total",
            "Total STT cost in USD",
            ["provider"],
            multiprocess_mode="liveall",
            registry=self._registry,
        )
        self.tts_cost = Gauge(
            "livekit_tts_cost_total",
            "Total TTS cost in USD",
            ["provider"],
            multiprocess_mode="liveall",
            registry=self._registry,
        )

        for metric in [
            self.llm_tokens,
            self.stt_duration,
            self.tts_chars,
            self.conversation_turns,
            self.total_tokens,
        ]:
            metric._multiprocess_mode = "livesum"  # type: ignore[reportPrivateUsage]

    def initialize_metrics(self) -> None:
        """Initialize metrics with default labels to ensure they exist."""
        logger.debug("Initializing metrics with default values...")
        cfg = self._config
        self.llm_latency.labels(
            model=cfg.primary_llm.model, agent_type=cfg.agent_type
        ).set(0)
        self.llm_latency_small.labels(
            model=cfg.fast_llm.model, agent_type=cfg.agent_type
        ).set(0)
        self.stt_latency.labels(
            provider=cfg.stt.provider, agent_type=cfg.agent_type
        ).set(0)
        self.tts_latency.labels(
            provider=cfg.tts.provider, agent_type=cfg.agent_type
        ).set(0)
        self.eou_latency.labels(agent_type=cfg.agent_type).set(0)
        self.total_conversation_latency.labels(agent_type=cfg.agent_type).set(0)

        self.llm_tokens.labels(type="prompt", model=cfg.primary_llm.model).inc(0)
        self.llm_tokens.labels(type="completion", model=cfg.primary_llm.model).inc(0)
        self.stt_duration.labels(provider=cfg.stt.provider).inc(0)
        self.tts_chars.labels(provider=cfg.tts.provider).inc(0)
        self.total_tokens.inc(0)

        self.llm_cost.labels(model=cfg.primary_llm.model).set(0)
        self.stt_cost.labels(provider=cfg.stt.provider).set(0)
        self.tts_cost.labels(provider=cfg.tts.provider).set(0)
        logger.debug("Successfully initialized all metrics.")

    def handle_event(self, ev: MetricsCollectedEvent) -> None:
        """Main event handler for all metrics events."""
        metrics.log_metrics(ev.metrics)
        self._update_usage_and_cost(ev.metrics)
        self._update_latency(ev)

    def _start_new_turn(self, room: str) -> None:
        self._turn_id_counter += 1
        self._current_turn_metrics = {
            "eou_delay": None,
            "llm_ttft": None,
            "tts_ttfb": None,
        }
        self.conversation_turns.labels(
            agent_type=self._config.agent_type, room=room
        ).inc()
        logger.debug(
            f"Started new turn with turn_id={self._turn_id_counter}, room={room}"
        )

    def _calculate_total_latency(self) -> None:
        if all(
            self._current_turn_metrics[k] is not None
            for k in ["eou_delay", "llm_ttft", "tts_ttfb"]
        ):
            eou_ms = self._current_turn_metrics["eou_delay"] * 1000  # pyright: ignore[reportOptionalOperand]
            llm_ms = self._current_turn_metrics["llm_ttft"] * 1000  # pyright: ignore[reportOptionalOperand]
            tts_ms = self._current_turn_metrics["tts_ttfb"] * 1000  # pyright: ignore[reportOptionalOperand]
            total_ms = int(eou_ms + llm_ms + tts_ms)

            logger.debug(
                f"Latency components (ms): EOU={int(eou_ms)}, "
                f"LLM={int(llm_ms)}, "
                f"TTS={int(tts_ms)}"
            )

            self.total_conversation_latency.labels(
                agent_type=self._config.agent_type
            ).set(total_ms)
            logger.info(
                "Updated total conversation latency metric",
                extra={
                    "current_value_ms": total_ms,
                    "timestamp": datetime.now().isoformat(),
                    "turn_id": self._turn_id_counter,
                },
            )

            logger.info(
                "Total Conversation Latency",
                extra={
                    "total_latency_ms": total_ms,
                    "eou_delay_ms": int(eou_ms),
                    "llm_ttft_ms": int(llm_ms),
                    "tts_ttfb_ms": int(tts_ms),
                    "timestamp": datetime.now().isoformat(),
                    "turn_id": self._turn_id_counter,
                },
            )
            # Reset for next turn calculation within the same session
            self._current_turn_metrics = {
                "eou_delay": None,
                "llm_ttft": None,
                "tts_ttfb": None,
            }

    def _update_usage_and_cost(self, m: AgentMetrics) -> None:
        """Update usage counters and cost gauges based on the latest summary."""
        self._usage_collector.collect(m)
        logger.debug(f"Usage metrics collected: {m}")
        summary = self._usage_collector.get_summary()
        logger.debug(f"Current usage summary: {summary}")

        # Calculate deltas
        prompt_tokens_delta = (
            summary.llm_prompt_tokens - self._last_usage_summary.llm_prompt_tokens
        )
        completion_tokens_delta = (
            summary.llm_completion_tokens
            - self._last_usage_summary.llm_completion_tokens
        )
        stt_duration_delta = (
            summary.stt_audio_duration - self._last_usage_summary.stt_audio_duration
        )
        tts_chars_delta = (
            summary.tts_characters_count - self._last_usage_summary.tts_characters_count
        )

        # Update Prometheus counters with deltas
        if prompt_tokens_delta > 0:
            self.llm_tokens.labels(
                type="prompt", model=self._config.primary_llm.model
            ).inc(prompt_tokens_delta)
        if completion_tokens_delta > 0:
            self.llm_tokens.labels(
                type="completion", model=self._config.primary_llm.model
            ).inc(completion_tokens_delta)
        if prompt_tokens_delta > 0 or completion_tokens_delta > 0:
            self.total_tokens.inc(prompt_tokens_delta + completion_tokens_delta)

        if stt_duration_delta > 0:
            self.stt_duration.labels(provider=self._config.stt.provider).inc(
                stt_duration_delta
            )
        if tts_chars_delta > 0:
            self.tts_chars.labels(provider=self._config.tts.provider).inc(
                tts_chars_delta
            )

        # Update cost gauges with cumulative values
        llm_cost = (
            summary.llm_prompt_tokens * self._config.primary_llm.cost_per_input_token
        ) + (
            summary.llm_completion_tokens
            * self._config.primary_llm.cost_per_output_token
        )
        stt_cost = summary.stt_audio_duration * self._config.stt.cost_per_second
        tts_cost = summary.tts_characters_count * self._config.tts.cost_per_character

        logger.info(
            "Cost calculation details",
            extra={
                "llm_tokens": {
                    "prompt_tokens": summary.llm_prompt_tokens,
                    "completion_tokens": summary.llm_completion_tokens,
                    "total_llm_cost": llm_cost,
                },
                "stt_duration": {
                    "seconds": summary.stt_audio_duration,
                    "cost": stt_cost,
                },
                "tts_chars": {"count": summary.tts_characters_count, "cost": tts_cost},
                "total_cost": llm_cost + stt_cost + tts_cost,
            },
        )

        self.llm_cost.labels(model=self._config.primary_llm.model).set(llm_cost)
        self.stt_cost.labels(provider=self._config.stt.provider).set(stt_cost)
        self.tts_cost.labels(provider=self._config.tts.provider).set(tts_cost)

        logger.info(
            "Updated cost metrics",
            extra={
                "llm_cost": llm_cost,
                "stt_cost": stt_cost,
                "tts_cost": tts_cost,
                "total_cost": llm_cost + stt_cost + tts_cost,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        self._last_usage_summary = summary

    def _update_latency(self, ev: MetricsCollectedEvent) -> None:
        """Update latency gauges based on specific metric events."""
        m = ev.metrics
        cfg = self._config

        if isinstance(m, EOUMetrics):
            logger.debug(f"Processing EOU metrics: {m}")
            self._start_new_turn(room=getattr(ev, "room", "unknown"))
            delay_ms = m.end_of_utterance_delay * 1000
            logger.debug(f"Observed EOU delay: {delay_ms}ms")
            self.eou_latency.labels(agent_type=cfg.agent_type).set(delay_ms)
            self._current_turn_metrics["eou_delay"] = m.end_of_utterance_delay
            self._calculate_total_latency()
            logger.info(
                "EOU Metrics",
                extra={
                    "end_of_utterance_delay": round(m.end_of_utterance_delay, 2),
                    "transcription_delay": round(m.transcription_delay, 2),
                    "on_user_turn_completed_delay": round(
                        m.on_user_turn_completed_delay, 2
                    ),
                    "speech_id": m.speech_id,
                    "timestamp": datetime.now().isoformat(),
                    "turn_id": self._turn_id_counter,
                },
            )
        elif isinstance(m, LLMMetrics):
            logger.debug(f"Processing LLM metrics: {m}")
            duration_ms = getattr(m, "duration", 0) * 1000
            if hasattr(m, "duration"):
                logger.debug(
                    f"Observed LLM response generation latency: {duration_ms}ms"
                )
            if hasattr(m, "ttft"):
                self.llm_latency.labels(
                    model=cfg.primary_llm.model, agent_type=cfg.agent_type
                ).set(m.ttft * 1000)
                self._current_turn_metrics["llm_ttft"] = m.ttft
                self._calculate_total_latency()
            logger.info(
                "LLM Metrics",
                extra={
                    "latency_ms": round(duration_ms, 2),
                    "total_tokens": getattr(m, "total_tokens", 0),
                    "timestamp": datetime.now().isoformat(),
                    "turn_id": self._turn_id_counter,
                },
            )
        elif isinstance(m, TTSMetrics):
            logger.debug(f"Processing TTS metrics: {m}")
            duration_ms = getattr(m, "duration", 0) * 1000
            if hasattr(m, "duration"):
                logger.debug(f"Observed TTS latency: {duration_ms}ms")
            if hasattr(m, "ttfb"):
                self.tts_latency.labels(
                    provider=cfg.tts.provider, agent_type=cfg.agent_type
                ).set(m.ttfb * 1000)
                self._current_turn_metrics["tts_ttfb"] = m.ttfb
                self._calculate_total_latency()
            logger.info(
                "TTS Metrics",
                extra={
                    "latency_ms": round(duration_ms, 2),
                    "timestamp": datetime.now().isoformat(),
                    "turn_id": self._turn_id_counter,
                },
            )
        elif isinstance(m, STTMetrics):
            logger.debug(f"Processing STT metrics: {m}")
            duration_ms = getattr(m, "duration", 0) * 1000
            if hasattr(m, "duration"):
                self.stt_latency.labels(
                    provider=cfg.stt.provider, agent_type=cfg.agent_type
                ).set(duration_ms)
                logger.debug(
                    f"Observed STT latency to generate transcript: {duration_ms}ms"
                )
            logger.info(
                "STT Metrics",
                extra={
                    "latency_ms": round(duration_ms, 2),
                    "timestamp": datetime.now().isoformat(),
                },
            )
        elif isinstance(m, VADMetrics):
            pass
            # logger.debug(f"Processing VAD metrics: {m}")
            # logger.info(
            #     "VAD Metrics",
            #     extra={
            #         "metrics": str(m),
            #         "timestamp": datetime.utcnow().isoformat(),
            #     },
            # )
        else:
            logger.debug(f"Received unknown metrics type: {type(m)}")

    def session_started(self) -> None:
        self.active_conversations.labels(agent_type=self._config.agent_type).inc()

    def decrement_active_conversations(self) -> None:
        self.active_conversations.labels(agent_type=self._config.agent_type).dec()

    async def log_session_summary(self) -> None:
        summary = self._usage_collector.get_summary()
        summary_dict = {
            "llm_prompt_tokens": summary.llm_prompt_tokens,
            "llm_completion_tokens": summary.llm_completion_tokens,
            "stt_audio_duration": round(summary.stt_audio_duration, 2),
            "tts_characters_count": summary.tts_characters_count,
        }
        logger.info(
            "Session Summary",
            extra={
                "usage_summary": json.dumps(summary_dict),
                "timestamp": datetime.now().isoformat(),
            },
        )


# --- Agent Logic (Uses Dependency Injection) ---
class PreResponseAgent(Agent):
    def __init__(
        self,
        config: AppConfig,
        metrics_mgr: MetricsManager,
        primary_llm: llm.LLM,
        fast_llm: llm.LLM,
    ):
        super().__init__(
            instructions=config.agent_instructions,
            llm=primary_llm,
            allow_interruptions=config.allow_interruptions,
        )
        self._config = config
        self._metrics_mgr = metrics_mgr
        self._fast_llm = fast_llm
        self._fast_llm_prompt = llm.ChatMessage(
            role="system",
            content=[config.fast_llm_prompt],
        )

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ):
        fast_llm_ctx = turn_ctx.copy(
            exclude_instructions=True, exclude_function_call=True
        ).truncate(max_items=3)
        fast_llm_ctx.items.insert(0, self._fast_llm_prompt)
        fast_llm_ctx.items.append(new_message)

        fast_llm_fut = asyncio.Future[str]()

        async def _fast_llm_reply() -> AsyncIterable[str]:
            filler_response = ""
            start_time = time.time()
            ttfb_recorded = False
            async for chunk in self._fast_llm.chat(
                chat_ctx=fast_llm_ctx
            ).to_str_iterable():
                if not ttfb_recorded:
                    ttfb = (time.time() - start_time) * 1000
                    self._metrics_mgr.llm_latency_small.labels(
                        model=self._config.fast_llm.model,
                        agent_type=self._config.agent_type,
                    ).set(ttfb)
                    logger.info(
                        "Fast LLM TTFB",
                        extra={
                            "ttfb_ms": ttfb,
                            "model": self._config.fast_llm.model,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )
                    ttfb_recorded = True
                filler_response += chunk
                yield chunk

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            logger.info(
                "Fast LLM response total duration",
                extra={
                    "duration_ms": duration_ms,
                    "model": self._config.fast_llm.model,
                    "response": filler_response,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
            fast_llm_fut.set_result(filler_response)

        self.session.say(_fast_llm_reply(), add_to_chat_ctx=False)
        filler_response = await fast_llm_fut
        logger.info(f"Fast response: {filler_response}")
        turn_ctx.add_message(
            role="assistant", content=filler_response, interrupted=False
        )


async def pre_warmup_test(llm: llm.LLM, tts: tts.TTS, stt: stt.STT) -> None:
    # LLM check

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Say hello")

    response_text = ""
    async with llm.chat(chat_ctx=chat_ctx) as stream:
        async for chunk in stream:
            response_text += str(chunk)

    if "hello" not in response_text.lower():
        raise ValueError("LLM test failed: unexpected response")
    logger.debug(f"LLM test response: {response_text}")

    # TTS check
    tts_audio = []
    async with tts.synthesize(response_text) as stream:
        audio = await stream.collect()
        tts_audio.append(audio)

    if len(tts_audio) == 0:
        raise ValueError("TTS test failed: no audio generated")

    logger.debug(f"TTS test audio length: {sum(len(a) for a in tts_audio)} bytes")

    # STT check
    stt_result = await stt.recognize(tts_audio)

    if "hello" not in stt_result.alternatives:
        raise ValueError("STT test failed: unexpected transcription")
    logger.debug(f"STT test transcription: {stt_result.alternatives}")


# --- Application Entrypoint (Composition Root) ---
async def entrypoint(ctx: JobContext):
    if not ctx.proc.userdata.get("vad", None):
        raise ValueError("VAD plugin not found in process userdata")

    config = AppConfig()
    logger.info("Loaded application config", extra={"config": config.model_dump()})

    metrics_mgr = MetricsManager(config)
    plugin_registry = PluginRegistry()

    primary_llm = plugin_registry.create_llm(config.primary_llm)
    fast_llm = plugin_registry.create_llm(config.fast_llm)
    stt_plugin = plugin_registry.create_stt(config.stt)
    tts_plugin = plugin_registry.create_tts(config.tts)
    vad_plugin = ctx.proc.userdata["vad"]

    agent = PreResponseAgent(
        config=config,
        metrics_mgr=metrics_mgr,
        primary_llm=primary_llm,
        fast_llm=fast_llm,
    )

    session = AgentSession(
        stt=stt_plugin,
        tts=tts_plugin,
        vad=vad_plugin,
        preemptive_generation=True,
        # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
        # when it's detected, you may resume the agent's speech
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
        min_interruption_duration=0.2,  # with false interruption resume, interruption can be more sensitive
    )

    session.on("metrics_collected", metrics_mgr.handle_event)
    metrics_mgr.session_started()
    atexit.register(metrics_mgr.decrement_active_conversations)
    ctx.add_shutdown_callback(metrics_mgr.log_session_summary)

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    await session.start(
        agent,
        room=ctx.room,
    )

    await asyncio.sleep(0.7)
    await session.say("Hi there, how are you doing today?", allow_interruptions=True)


def prewarm(proc: JobProcess):
    config = AppConfig()
    logger.info("Loaded application config", extra={"config": config.model_dump()})

    # plugin_registry = PluginRegistry()

    # primary_llm = plugin_registry.create_llm(config.primary_llm)

    # stt_plugin = plugin_registry.create_stt(config.stt)
    # tts_plugin = plugin_registry.create_tts(config.tts)
    # try:
    #     _ = asyncio.run(pre_warmup_test(primary_llm, tts_plugin, stt_plugin))
    # except Exception as e:
    #     logger.error(f"Pre-warmup test failed: {e}", exc_info=True)
    #     raise
    proc.userdata["vad"] = silero.VAD.load(**config.vad.model_dump())


if __name__ == "__main__":
    try:
        main_config = AppConfig()
        main_metrics_mgr = MetricsManager(main_config)
        main_metrics_mgr.initialize_metrics()

        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True)
