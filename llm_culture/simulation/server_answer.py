import os

import httpx
from openai import OpenAI


# Appended to the raw prompt when `instruct=False`, to coax a base (non-chat)
# model into continuing as the assistant.
_COMPLETION_PRIMER = "Assistant: Sure, here is the requested answer:\n\n1."


# --- vLLM helpers (shared by the single-prompt and batch paths) ---------------

def _vllm_conversations(tokenizer, prompts, instruct):
    """Render each prompt into the string vLLM expects (chat-templated or primed)."""
    if instruct:
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False
            )
            for p in prompts
        ]
    return [p + _COMPLETION_PRIMER for p in prompts]


def _vllm_sampling_params(tokenizer, generation, sampling_params):
    """Use the caller's SamplingParams if given, else build one from `generation`."""
    if sampling_params is not None:
        return sampling_params
    from vllm import SamplingParams  # optional "serving" extra

    return SamplingParams(
        temperature=generation.temperature,
        top_p=generation.top_p,
        max_tokens=generation.max_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
    )


# --- single-prompt backends ---------------------------------------------------

def _generate_vllm(model, prompt, generation, instruct, sampling_params):
    """In-process vLLM generation (GPU)."""
    tokenizer = model.get_tokenizer()
    conversations = _vllm_conversations(tokenizer, [prompt], instruct)
    params = _vllm_sampling_params(tokenizer, generation, sampling_params)
    output = model.generate(conversations, params)
    return output[0].outputs[0].text


def _generate_llama_cpp(model, prompt, generation, instruct):
    """In-process llama.cpp generation (CPU / Apple Metal)."""
    if instruct:
        output = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=generation.temperature,
            top_p=generation.top_p,
            max_tokens=generation.max_tokens,
        )
        return output["choices"][0]["message"]["content"]

    output = model(
        prompt + _COMPLETION_PRIMER,
        temperature=generation.temperature,
        top_p=generation.top_p,
        max_tokens=generation.max_tokens,
    )
    return output["choices"][0]["text"]


def _build_remote_client(access_url, verify, timeout, max_retries):
    """OpenAI-compatible client; the SDK handles retries/backoff/timeouts."""
    base_url = access_url.rstrip("/") + "/v1"
    client_kwargs = dict(
        base_url=base_url,
        # Local servers (vLLM / llama.cpp) accept any key; env override for hosted.
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        timeout=timeout,
        max_retries=max_retries,
    )
    if not verify:
        # Disabling TLS verification requires injecting a custom http client.
        client_kwargs["http_client"] = httpx.Client(verify=False, timeout=timeout)
    return OpenAI(**client_kwargs), base_url


def _generate_remote(
    access_url, prompt, generation, instruct, start_flag, model,
    debug, verify, timeout, max_retries,
):
    """Generation via a remote OpenAI-compatible server (HTTP)."""
    client, base_url = _build_remote_client(access_url, verify, timeout, max_retries)

    # Local servers ignore the model name; env override for servers that validate it.
    request_model = model if isinstance(model, str) else os.environ.get("OPENAI_MODEL", "local-model")

    if debug:
        print("POST base_url:", base_url, "| instruct:", instruct, "| model:", request_model)

    try:
        if instruct:
            response = client.chat.completions.create(
                model=request_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=generation.temperature,
                top_p=generation.top_p,
                max_tokens=generation.max_tokens,
            )
            return response.choices[0].message.content.replace("</s>", "")

        prompt = prompt + _COMPLETION_PRIMER
        if start_flag is not None:
            prompt += start_flag

        response = client.completions.create(
            model=request_model,
            prompt=prompt,
            temperature=generation.temperature,
            top_p=generation.top_p,
            max_tokens=generation.max_tokens,
        )
        return response.choices[0].text
    except Exception as exc:
        # Surface a clear error instead of hanging or leaking a raw client error.
        raise RuntimeError(
            f"LLM request to {base_url} failed after {max_retries} retries"
        ) from exc


def get_answer(
    access_url,
    prompt,
    generation,
    debug=False,
    instruct=True,
    start_flag=None,
    llm_backend=False,
    model=None,
    sampling_params=None,
    verify=True,
    timeout=120,
    max_retries=5,
):
    """Generate one answer, dispatching to the selected backend."""
    if llm_backend == "vllm":
        return _generate_vllm(model, prompt, generation, instruct, sampling_params)
    if llm_backend == "llama.cpp":
        return _generate_llama_cpp(model, prompt, generation, instruct)
    return _generate_remote(
        access_url, prompt, generation, instruct, start_flag, model,
        debug, verify, timeout, max_retries,
    )


# --- batch backends -----------------------------------------------------------

def _generate_vllm_batch(model, prompts, generation, instruct, sampling_params):
    """One native batched vLLM call; outputs come back in input order."""
    tokenizer = model.get_tokenizer()
    conversations = _vllm_conversations(tokenizer, prompts, instruct)
    params = _vllm_sampling_params(tokenizer, generation, sampling_params)
    outputs = model.generate(conversations, params)
    return [o.outputs[0].text for o in outputs]


def _generate_sequential(prompts, one):
    """Run prompts one after another (llama.cpp high-level binding is single-sequence)."""
    return [one(p) for p in prompts]


def _generate_concurrent(prompts, one, max_concurrent_requests):
    """Fire independent requests concurrently; the server overlaps them.

    ThreadPoolExecutor.map keeps input order and re-raises the first exception.
    """
    if max_concurrent_requests <= 1 or len(prompts) == 1:
        return _generate_sequential(prompts, one)
    from concurrent.futures import ThreadPoolExecutor

    workers = min(max_concurrent_requests, len(prompts))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, prompts))


def get_answers_batch(
    prompts,
    generation,
    *,
    access_url="",
    debug=False,
    instruct=True,
    llm_backend=False,
    model=None,
    sampling_params=None,
    max_concurrent_requests=8,
    verify=True,
    timeout=120,
    max_retries=5,
):
    """Generate one answer per prompt, batched where the backend supports it.

    Returns a list aligned 1:1 (and in order) with `prompts`. Prompts in one call
    must be INDEPENDENT (a single simulation timestep). Per backend:
      * "vllm"      -> one native batched model.generate(list) (continuous batching)
      * "llama.cpp" -> sequential loop (the high-level binding is single-sequence)
      * remote      -> concurrent requests; the server's batching does the overlap
    """
    if not prompts:
        return []

    if llm_backend == "vllm":
        return _generate_vllm_batch(model, prompts, generation, instruct, sampling_params)

    def _one(prompt):
        # Exactly one answer via the single-call path (shares its retries).
        return get_answer(
            access_url,
            prompt,
            generation,
            debug=debug,
            instruct=instruct,
            llm_backend=llm_backend,
            model=model,
            sampling_params=sampling_params,
            verify=verify,
            timeout=timeout,
            max_retries=max_retries,
        )

    if llm_backend == "llama.cpp":
        # For real batching here, run the llama.cpp *server* with
        # --parallel/--cont-batching and use the remote path instead.
        return _generate_sequential(prompts, _one)

    return _generate_concurrent(prompts, _one, max_concurrent_requests)
