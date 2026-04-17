"""End-to-end voice-agent demo against ``lunavox serve``.

Purpose
-------
Phase 5C shipped the ``WS /v1/stream/text`` endpoint — a
sentence-streaming **input** channel for voice agents. The server
takes text frames one at a time, watches for sentence boundaries, and
starts emitting audio PCM as soon as the first complete sentence
lands. That's how you get LLM-driven TTS with ~one-sentence latency
instead of "full LLM reply + first sentence TTFB" latency.

This script fakes the LLM side so you can see the pipeline work
without wiring up OpenAI / llama.cpp / Ollama. It:

1. Connects to ``ws://HOST:PORT/v1/stream/text`` on a running
   ``lunavox serve`` instance.
2. Sends the init frame (voice config).
3. Emits a scripted story word-by-word with a configurable per-word
   delay that mimics LLM token streaming latency.
4. Receives binary PCM frames as they arrive and writes them to a
   WAV file.
5. Receives the terminal JSON (sentence count, per-sentence stats).
6. Prints a timing summary:
     * first-audio TTFB from the first text frame
     * total wall time from start to stream close
     * sentences detected by the server
     * ratio of audio duration to wall time (lower = better agent UX)

Prerequisites
-------------
In one terminal::

    pip install lunavox
    lunavox model pull --model base_small
    lunavox build
    lunavox serve --batch-size 2 --port 8765

In another terminal::

    pip install websockets
    python examples/voice_agent_demo.py --port 8765 \\
        --word-delay-ms 40 --output output/agent_demo.wav

Swap the fake LLM for a real one by pointing the script at an
``AsyncIterator[str]``. The :func:`_fake_llm_tokens` function below
is the only piece that assumes a scripted source.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

try:
    import websockets
except ImportError:
    print(
        "This demo needs the `websockets` package. Install it with:",
        "  pip install websockets",
        sep="\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from None


# Scripted reply — 6 sentences, mix of short + medium + long so
# the per-sentence synthesis pattern is visible in the timing output.
_FAKE_LLM_REPLY = (
    "Hello there. "
    "I am a voice agent demonstration. "
    "This message is being streamed to the synthesizer "
    "one word at a time. "
    "By the time you hear the first sentence, the rest are still "
    "being produced in parallel. "
    "Thank you for listening!"
)

# Longer multilingual reply for stress-testing the Phase 6 auto-split
# pipeline. Each individual sentence is within normal bounds, but the
# total exceeds the pipeline's ``auto_split_threshold`` many times
# over — so the server must split, synthesize per segment, and stitch
# PCM chunks back together transparently. English + Chinese + Japanese
# mixed so the multi-language punctuation table gets exercised end-to-end.
_LONG_MULTILINGUAL_REPLY = (
    "Welcome to the LunaVox long-text demo. "
    "This extended reply exercises the automatic sentence splitter. "
    "It mixes several languages on purpose. "
    "你好,世界。这是一个中文句子,测试全角标点的自动切分能力。"
    "今天天气很好,我们一起散步吧!"
    "こんにちは、世界。これは日本語の文章です。"
    "長い文章でも、句読点に沿って綺麗に切り分けられるはずです。"
    "Back to English now. "
    "The pipeline ensures that chunks arrive in order across every "
    "segment, and only the very last audio chunk carries the "
    "end-of-stream signal. "
    "That means downstream consumers see one clean stream regardless "
    "of how many internal splits happened on the server side. "
    "最后用一句中文结束整段演示。"
)


async def _fake_llm_tokens(
    reply: str = _FAKE_LLM_REPLY,
    word_delay_ms: int = 40,
) -> AsyncIterator[str]:
    """Yield ``reply`` word-by-word with ``word_delay_ms`` between each.

    The delay mimics an LLM streaming tokens at ~25 tokens/second.
    Whitespace is preserved so the server's
    :class:`~lunavox.core.text.StreamingSentenceBuffer` sees boundary
    cues (terminator + space) at the right moments.
    """
    delay = word_delay_ms / 1000.0
    # Split on spaces but keep the spaces so the receiver can stitch
    # the sentence back together.
    parts = reply.split(" ")
    for i, word in enumerate(parts):
        chunk = word + (" " if i < len(parts) - 1 else "")
        yield chunk
        await asyncio.sleep(delay)


def _write_pcm_chunks_as_wav(
    path: Path,
    chunks: list[bytes],
    sample_rate: int,
) -> int:
    """Stitch int16-LE PCM chunks into a single mono WAV file.

    Returns the total number of samples written so callers can
    double-check against the server-reported count.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joined = b"".join(chunks)
    n_samples = len(joined) // 2  # 16-bit
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(joined)
    return n_samples


async def run_demo(args: argparse.Namespace) -> int:
    url = f"ws://{args.host}:{args.port}/v1/stream/text"
    print(f"→ connecting to {url}")

    init_frame = {
        "voice": args.voice,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.reference:
        init_frame["reference"] = args.reference
    if args.speaker:
        init_frame["speaker"] = args.speaker
    if args.instruct:
        init_frame["instruct"] = args.instruct

    first_text_sent_at: float | None = None
    first_audio_at: float | None = None
    pcm_chunks: list[bytes] = []
    terminal: dict[str, Any] | None = None

    reply_text = _LONG_MULTILINGUAL_REPLY if args.long else _FAKE_LLM_REPLY

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps(init_frame))

        async def _producer() -> None:
            nonlocal first_text_sent_at
            async for token in _fake_llm_tokens(
                reply=reply_text, word_delay_ms=args.word_delay_ms
            ):
                if first_text_sent_at is None:
                    first_text_sent_at = time.perf_counter()
                await ws.send(json.dumps({"text": token}))
            await ws.send(json.dumps({"end": True}))

        async def _consumer() -> None:
            nonlocal first_audio_at, terminal
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    if first_audio_at is None:
                        first_audio_at = time.perf_counter()
                    pcm_chunks.append(bytes(msg))
                else:
                    frame = json.loads(msg)
                    if frame.get("error"):
                        print(f"server error: {frame['error']}", file=sys.stderr)
                        return
                    if frame.get("done"):
                        terminal = frame
                        return

        t0 = time.perf_counter()
        await asyncio.gather(_producer(), _consumer())
        t_total = time.perf_counter() - t0

    if terminal is None:
        print("stream closed without a terminal frame", file=sys.stderr)
        return 1

    sample_rate = int(terminal.get("sample_rate") or 24000)
    n_samples_written = _write_pcm_chunks_as_wav(Path(args.output), pcm_chunks, sample_rate)
    audio_seconds = n_samples_written / sample_rate

    ttfb_ms = (
        (first_audio_at - first_text_sent_at) * 1000.0
        if first_audio_at and first_text_sent_at
        else -1.0
    )

    print()
    print("=" * 60)
    print(f"  WAV written       : {args.output}")
    print(f"  audio samples     : {n_samples_written:,}")
    print(f"  audio seconds     : {audio_seconds:.2f}")
    print(f"  sentences         : {terminal.get('sentences')}")
    print(f"  first-audio TTFB  : {ttfb_ms:.0f} ms  (from first text chunk → first PCM frame)")
    print(f"  total wall time   : {t_total * 1000:.0f} ms")
    print(
        f"  audio / wall      : {audio_seconds / t_total:.2f}×  "
        f"(>1 means the agent is faster than real-time talking)"
    )
    if terminal.get("stats"):
        s = terminal["stats"]
        print(
            f"  last-sentence     : rtf={s.get('rtf', 0):.3f}  "
            f"total_ms={s.get('t_total_ms', 0)}  "
            f"duration_ms={s.get('audio_duration_ms', 0)}"
        )
    print("=" * 60)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("Prerequisites")[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--word-delay-ms",
        type=int,
        default=40,
        help="Inter-word delay while streaming the fake LLM reply. "
        "40 ms ≈ 25 tokens/sec, a realistic local-LLM pace.",
    )
    parser.add_argument(
        "--output",
        default="output/agent_demo.wav",
        help="WAV file to write the synthesized audio to.",
    )
    parser.add_argument(
        "--voice",
        default="base",
        choices=("base", "clone", "custom", "design"),
        help="Voice mode. For non-base, also pass --reference / --speaker / --instruct.",
    )
    parser.add_argument("--reference", default=None)
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    parser.add_argument(
        "--long",
        action="store_true",
        help="Use the long multilingual reply instead of the short demo "
        "script. Exercises the Phase 6 auto-split pipeline end-to-end: "
        "total input well over the server's split threshold, mixed "
        "EN/CJK punctuation, chunk ordering preserved across segments.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(run_demo(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
