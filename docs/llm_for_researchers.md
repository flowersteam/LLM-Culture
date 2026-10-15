# Running an LLM: a plain-language primer

The LLM concepts behind LLM-Culture, for anyone who hasn't run a language model before.
No machine-learning background assumed. The goal is that the words in the config
files — *backend*, *GGUF*, *quantization*, *context window*, *temperature* — stop being
magic, so you can make sensible choices for your hardware and your runs.

You do not need any of this for the quickstart. Read it when you want to run a *real*
model, understand a setting, or decide where to run your experiment.

---

## 1. What the model is actually doing

In LLM-Culture the model is the thing that writes the text. An agent is shown the
stories of its network neighbours and asked, via its prompts, to write a new one — the
model is what turns "here are some stories, now write one" into actual text. Everything
below is about *which* model does that and *how* it writes.

**Tokens.** Models don't read characters or words; they read **tokens** — chunks of
text, roughly ¾ of a word on average ("transmission" might be 2–3 tokens). This matters
for two practical reasons: length limits are counted in tokens (not words), and a
model's speed is measured in **tokens per second**.

**Instruct vs. base models.** A **base** model only continues text. An **instruct**
(a.k.a. chat) model has been further trained to follow instructions like "write a
story". For this framework you almost always want an **instruct** model — the prompts
are instructions. Model names usually say so (e.g. `Mistral-7B-Instruct`). This is what
the `generation.instruct` setting toggles (`true` = treat it as a chat/instruct model,
the default).

---

## 2. Where the model runs (inference / "backend")

"Running" a model (the technical word is **inference**) means loading its parameters
into memory and doing the computation that produces text. The **backend** is the engine
that does this. LLM-Culture supports three, and the only real question is **where the
computation happens**:

| `backend.kind` | Where | Good for | Cost |
| --- | --- | --- | --- |
| `llama_cpp` | your own computer (CPU or Apple-Silicon GPU) | piloting, teaching, small studies | free |
| `vllm` | a Linux machine with an NVIDIA GPU | real data collection — big models, fast | your GPU/cluster |
| `none` | a remote server you call over the network (a hosted API, or a server you started) | best quality with no local hardware | usually per-token $$ + an API key |

How to choose:

- **Just trying it out, or no GPU?** Use `llama_cpp` on your own machine. This is what
  the quickstart does. It runs anywhere, but larger models are slow on a laptop.
- **Collecting real data and you have a GPU box (or a cluster/Colab)?** Use `vllm`. It
  is built for running a model many times quickly, which is exactly what a population of
  agents over many generations needs.
- **Want a strong model and don't want to manage hardware?** Use a hosted API
  (`none` + a URL). You pay per use and need a key, but there's nothing to install.

The ready-made presets `experiment=mac_mx` (Apple Silicon) and `experiment=linux_gpu`
(Linux GPU) are concrete, working answers for the first two cases.

### Will the model fit on my machine?

A model's size is given by its **parameter count** — 135M, 7B ("B" = billion), 70B, …
More parameters generally means better text but more memory and slower generation. The
practical question is whether the model **fits in memory**. A rough rule:

> memory needed ≈ (model parameters) × (bytes per parameter) + a bit for working space

At full precision a parameter takes 2 bytes, so a 7B model needs ~14 GB just to load —
too much for most laptops. This is where **quantization** comes in.

### Quantization and GGUF (how a big model fits on a small machine)

**Quantization** stores each parameter in fewer bits, shrinking the model (with a small
quality cost). Instead of 16 bits per parameter you might use ~4. A 7B model then needs
~4–5 GB instead of ~14 GB — small enough for a laptop.

- **GGUF** is the file format `llama_cpp` uses. A single `.gguf` file is a packaged,
  quantized model. On Hugging Face a model's "GGUF repo" usually contains *several*
  `.gguf` files, one per quantization level — you pick one with
  `backend.llama_cpp.gguf_filename`.
- **Q4_K_M** is the most common sweet spot: 4-bit quantization with good quality for the
  size. The names form a ladder from smaller/rougher to larger/better:
  `Q2_K` → `Q4_K_M` → `Q6_K` → `Q8_0`. Start at `Q4_K_M`; go up if you have memory to
  spare and want better text, down if you're tight.
- For `vllm` you usually load the normal (non-GGUF) model and, if needed, use a
  GPU-oriented quantization (`awq` / `gptq`) via `backend.vllm.quantization`.

### Context window

The **context window** (`n_ctx` for llama.cpp, `max_model_len` for vLLM) is how many
**tokens** the model can look at in one go — the prompt (all the neighbour stories plus
instructions) **plus** the story it generates. If it is too small the run **crashes**
with a "requested tokens exceed context window" error.

Sizing it is not guesswork, because an agent's prompt contains *every* neighbour's
story:

> tokens needed ≈ (number of neighbours + 1) × `generation.max_tokens`

The `+ 1` is the story being written. In a `fully_connected` population every agent has
`n_agents - 1` neighbours, which is the demanding case: 10 agents at `max_tokens: 512`
needs ≈ 10 × 512 ≈ 5k tokens, so `4096` is **not** enough — use `8192`. A `sequence`
chain has a single neighbour and needs very little. If a run crashes, either raise the
context window or lower `generation.max_tokens`; the product of the two is what matters.

**In words.** Measured on this framework's own output, a token is ~0.77 words (1.3
tokens per word), so:

| tokens | ≈ words |
| --- | --- |
| `max_tokens: 256` | ~200 words (a long paragraph) |
| `max_tokens: 512` | ~390 words (a page) |
| `n_ctx: 4096` | ~3,100 words |
| `n_ctx: 8192` | ~6,300 words |

So the 10-agent example is asking each agent to read ~3,500 words of neighbour stories
before writing its own ~390 — which is why it needs a big window.

**The memory cost, and when it is paid.** The memory a context window needs is called
the **KV cache**, and the two backends treat it differently:

- **llama.cpp reserves it up front.** The whole `n_ctx` worth of KV cache is allocated
  when the model loads, whether or not your prompts ever get that long. So `n_ctx` is a
  direct, immediate memory cost: Mistral-7B needs ~128 KB per token, i.e. ~0.5 GB at
  4096 and ~1 GB at 8192. (At SmolLM2-135M scale it is only tens of MB.) This is why
  lowering `n_ctx` is the first thing to try when you run out of memory.
- **vLLM reserves a pool up front, then pages out of it.** At startup it claims
  `gpu_memory_utilization` of your VRAM as one big KV-cache pool, and each sequence is
  given 16-token blocks from that pool on demand. `max_model_len` is therefore a *cap*
  per sequence, not a preallocation — raising it does not by itself consume more memory,
  though vLLM refuses to start if the pool cannot hold at least one sequence of that
  length. The lever that controls vLLM's footprint is `gpu_memory_utilization`.

Note you cannot raise the context window indefinitely: each model has a maximum it was
trained for (8192 for SmolLM2, 32768 for Mistral-7B).

### Where models are stored

When you give a Hugging Face repo id (e.g. `mistralai/Mistral-7B-Instruct-v0.2`), it is
downloaded once to `~/.cache/huggingface` and reused afterwards. The first run of a new
model is slow because of the download; later runs are not. You can point
`backend.hf_cache_dir` elsewhere if your home disk is small.

### Using a hosted / remote API (`backend.kind=none`)

Point `backend.access_url` at any OpenAI-compatible endpoint (a commercial API, or a
vLLM / llama.cpp server you started yourself). Two optional environment variables cover
authenticated or model-validating servers:

- `OPENAI_API_KEY` — your auth token. Local servers accept the default (`EMPTY`); set a
  real key for a genuine hosted API. It is read from the environment — never hardcode it
  in a config.
- `OPENAI_MODEL` — the model name to request, for servers that validate it (local
  servers ignore it and serve whatever they loaded).

`backend.max_concurrent_requests` caps how many requests are sent at once when a
generation is run in parallel (remote only).

```bash
export OPENAI_API_KEY=sk-...          # only for a hosted/authenticated server
export OPENAI_MODEL=gpt-4o-mini       # only if the server validates the name
uv run python run_experiment.py backend.kind=none backend.access_url=https://my-server:8000
```

---

## 3. How the model writes (generation / sampling)

A model produces text one token at a time: at each step it assigns a probability to
every possible next token, and a **sampling** rule picks one. The `generation.*`
settings control that rule — and the same values apply to **every** backend, so you can
tune generation once and switch where the model runs freely.

- **`temperature`** — how random each choice is. At `0` the model always takes the most
  likely token, so it stays close to its input and is near-deterministic. Higher values
  flatten the probabilities so less-likely tokens get picked more often, adding variation
  to each rewrite. `0.7–1.0` is a typical range; above ~1.3 text starts to degrade.

- **`top_p`** (nucleus sampling) — a safety rail on randomness. The model considers only
  the most probable tokens whose probabilities add up to `p` (e.g. `0.95` = the top 95%
  of the probability mass), ignoring the long tail of very unlikely tokens. It mostly
  interacts with `temperature`; the default (`0.95`) is fine to leave alone.

- **`max_tokens`** — the longest a generated story may be, in tokens (not words). Caps
  runaway length and bounds how long each generation takes. Remember the context window
  must fit the prompt **plus** these output tokens.

- **`instruct`** — whether to use the model's chat/instruction interface (see §1). Leave
  `true` for instruct models (the usual case); set `false` only for a raw base model.
