# LunaVox Examples

Self-contained scripts that exercise LunaVox from the outside — the
same way a real integration would. Each example is a single file
with no build step; each assumes `lunavox` is installed
(`pip install "lunavox[serve]"` covers the serving-layer demos).

## Index

| File | What it shows | Requires |
| :--- | :--- | :--- |
| [`voice_agent_demo.py`](voice_agent_demo.py) | Streaming text **into** `lunavox serve` word-by-word the way an LLM would, and getting audio back sentence-by-sentence. Measures first-audio TTFB and prints per-sentence stats. | A running `lunavox serve`, the `websockets` pip package |

## Running `voice_agent_demo.py`

In one terminal:

```bash
pip install "lunavox[serve]"
lunavox model pull --model base_small
lunavox build
lunavox serve --batch-size 2 --port 8765
```

In another terminal:

```bash
pip install websockets
python examples/voice_agent_demo.py --port 8765 --word-delay-ms 40 \
    --output output/agent_demo.wav
```

You should see a summary like:

```
============================================================
  WAV written       : output/agent_demo.wav
  audio samples     : 245,760
  audio seconds     : 10.24
  sentences         : 5
  first-audio TTFB  : 240 ms  (from first text chunk → first PCM frame)
  total wall time   : 3280 ms
  audio / wall      : 3.12×  (>1 means the agent is faster than real-time talking)
  last-sentence     : rtf=0.185  total_ms=420  duration_ms=2280
============================================================
```

The "first-audio TTFB" is the metric that matters for voice-agent
UX: it's the delay from the moment the user's LLM starts generating
a reply to the moment LunaVox has audio to play. The
`WS /v1/stream/text` endpoint exists specifically to make this
number small — without sentence-level streaming you'd pay
"full LLM reply time + first-sentence synth time" instead of "first
sentence LLM time + first-sentence synth time".

### Using a different voice mode

Clone from a reference audio file:

```bash
python examples/voice_agent_demo.py \
    --voice clone \
    --reference ref/ref_0.6B.json \
    --output output/agent_clone.wav
```

Custom speaker:

```bash
python examples/voice_agent_demo.py \
    --voice custom --speaker Vivian --instruct "Use an excited tone." \
    --output output/agent_custom.wav
```

Design from a description:

```bash
python examples/voice_agent_demo.py \
    --voice design --instruct "A warm, calm narrator voice." \
    --output output/agent_design.wav
```

### Plugging in a real LLM

The script's fake LLM is one function — `_fake_llm_tokens` — that
yields strings asynchronously. Replace it with any
`AsyncIterator[str]` producing your own source:

```python
async def openai_tokens() -> AsyncIterator[str]:
    client = AsyncOpenAI()
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Tell me a story."}],
        stream=True,
    )
    async for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta
```

The rest of the pipeline (sentence boundary detection, concurrent
synthesis, audio streaming) is handled by `lunavox serve`, so you
don't need to change anything else.
