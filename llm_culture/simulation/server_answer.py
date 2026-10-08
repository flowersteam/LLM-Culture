import os

import httpx
from openai import OpenAI


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
    temperature = generation.temperature
    max_tokens = generation.max_tokens
    top_p = generation.top_p

    if llm_backend == "vllm":
        # Deferred import: vllm is an optional ("serving") extra, absent in base installs.
        from vllm import SamplingParams

        if instruct:
            tokenizer = model.get_tokenizer()

            conversations = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
            )
        else:
            conversations = (
                prompt +
                "Assistant: Sure, here is the requested answer:\n\n1."
            )

        output = model.generate(
            [conversations],
            sampling_params if sampling_params is not None else SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
            )
        )

        return output[0].outputs[0].text
    if llm_backend == "llama.cpp":
        if instruct:
            conversations = [
                {"role": "user", "content": prompt}
            ]

            output = model.create_chat_completion(
                messages=conversations,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        else:
            conversations = (
                prompt +
                "Assistant: Sure, here is the requested answer:\n\n1."
            )

            output = model(
                conversations,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        return output["choices"][0]["text"] if not instruct else \
            output["choices"][0]["message"]["content"]
    # Remote / server backend: talk to an OpenAI-compatible endpoint (a hosted
    # server, a vLLM server, or a llama.cpp server) using the official `openai`
    # client. The client handles retries + exponential backoff + timeouts for us,
    # so we no longer hand-roll a request loop. The request fields and the way we
    # read the response are kept identical to the previous implementation, so the
    # output contract is unchanged.
    base_url = access_url.rstrip("/") + "/v1"

    client_kwargs = dict(
        base_url=base_url,
        # OpenAI-compatible local servers (vLLM / llama.cpp) accept any key; allow
        # overriding via env for genuine hosted endpoints.
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        timeout=timeout,
        max_retries=max_retries,
    )
    if not verify:
        # Mirror the previous requests(..., verify=False) behavior only when asked;
        # a custom http_client is what lets us disable TLS verification.
        client_kwargs["http_client"] = httpx.Client(verify=False, timeout=timeout)

    client = OpenAI(**client_kwargs)

    # The HTTP path historically sent no explicit model name (local servers ignore
    # it / use the one they loaded). Keep a harmless default; override via the
    # OPENAI_MODEL env var for servers that validate it (e.g. a remote vLLM).
    request_model = model if isinstance(model, str) else os.environ.get("OPENAI_MODEL", "local-model")

    if debug:
        print("POST base_url:", base_url, "| instruct:", instruct, "| model:", request_model)

    try:
        if instruct:
            response = client.chat.completions.create(
                model=request_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.replace("</s>", "")

        prompt = prompt + "Assistant: Sure, here is the requested answer:\n\n1."
        if start_flag is not None:
            prompt += start_flag

        response = client.completions.create(
            model=request_model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].text
    except Exception as exc:
        # Surface a clear error instead of hanging or leaking a raw client error.
        raise RuntimeError(
            f"LLM request to {base_url} failed after {max_retries} retries"
        ) from exc


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

    Returns a list aligned 1:1 (and in order) with `prompts`. The prompts in one
    call must be INDEPENDENT (they come from a single simulation timestep). Per
    backend:
      * "vllm"      -> one native batched model.generate(list) (continuous batching)
      * "llama.cpp" -> sequential loop (the high-level binding is single-sequence)
      * remote      -> concurrent requests; the server's batching does the overlap
    """
    if not prompts:
        return []

    if llm_backend == "vllm":
        # Deferred import: vllm is an optional ("serving") extra.
        from vllm import SamplingParams

        tokenizer = model.get_tokenizer()
        if instruct:
            conversations = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False
                )
                for p in prompts
            ]
        else:
            conversations = [
                p + "Assistant: Sure, here is the requested answer:\n\n1."
                for p in prompts
            ]
        params = sampling_params if sampling_params is not None else SamplingParams(
            temperature=generation.temperature,
            top_p=generation.top_p,
            max_tokens=generation.max_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
        )
        # vLLM schedules all sequences together and returns them in input order.
        outputs = model.generate(conversations, params)
        return [o.outputs[0].text for o in outputs]

    def _one(prompt):
        # Exactly one answer via the existing single-call path (shares its retries).
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
        # In-process high-level llama.cpp is single-sequence: no true batched decode,
        # so run them one after another (correct, just not faster). For real batching
        # on this backend, run the llama.cpp *server* with --parallel/--cont-batching
        # and use the remote path below.
        return [_one(p) for p in prompts]

    # Remote OpenAI-compatible server: fire the independent requests concurrently and
    # let the server overlap them. ThreadPoolExecutor.map keeps input order and
    # re-raises the first exception (fail-loud, like get_answer).
    if max_concurrent_requests <= 1 or len(prompts) == 1:
        return [_one(p) for p in prompts]
    from concurrent.futures import ThreadPoolExecutor

    workers = min(max_concurrent_requests, len(prompts))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, prompts))