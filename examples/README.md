# LunaVox Examples

Self-contained scripts that exercise LunaVox from the outside — each
is a single file with no build step and assumes `lunavox` is installed
(`pip install lunavox` already ships the serving layer).

## Index

| File | What it shows | Requires |
| :--- | :--- | :--- |
| [`voice_agent_demo.py`](voice_agent_demo.py) | Streams text **into** `lunavox serve` word-by-word (LLM-style) and gets audio back sentence-by-sentence; reports first-audio TTFB and per-sentence stats. | Running `lunavox serve`; `pip install websockets` |

## Running `voice_agent_demo.py`

Terminal 1:

```bash
pip install lunavox
lunavox model pull --model base_small
lunavox build
lunavox serve --batch-size 2 --port 8765
```

Terminal 2:

```bash
pip install websockets
python examples/voice_agent_demo.py --port 8765 --word-delay-ms 40 \
    --output output/agent_demo.wav
```

The script's `--help`, the module docstring, and the stats summary it
prints at the end cover every flag and metric — start there for
details. The "first-audio TTFB" it reports is the metric that matters
for voice-agent UX: the delay from the moment the upstream LLM starts
replying to the moment LunaVox has audio to play. The
`WS /v1/stream/text` endpoint exists specifically to keep this number
small; see [serve guide](../docs/en/guide/serve.md) for the protocol.

### Other voice modes

```bash
python examples/voice_agent_demo.py --voice clone --reference ref/ref_0.6B.json -o output/clone.wav
python examples/voice_agent_demo.py --voice custom --speaker Vivian --instruct "Use an excited tone." -o output/custom.wav
python examples/voice_agent_demo.py --voice design --instruct "A warm, calm narrator voice." -o output/design.wav
```

### Plugging in a real LLM

Replace `_fake_llm_tokens` with any `AsyncIterator[str]` — e.g. an
OpenAI streaming client yielding each delta. The rest of the pipeline
(sentence boundaries, concurrent synthesis, audio streaming) is owned
by `lunavox serve`, so nothing else changes.
