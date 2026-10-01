def get_answer(
    access_url,
    prompt,
    debug=False,
    instruct=True,
    start_flag=None,
    llm_backend=False,
    model=None,
    temperature=0.8, 
    sampling_params=None,
    verify=True,
    timeout=120,
    max_retries=5,
):
    if llm_backend == "vllm":
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
                top_p=0.95,
                max_tokens=512,
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
                top_p=0.95,
                max_tokens=512,
            )
        else:
            conversations = (
                prompt +
                "Assistant: Sure, here is the requested answer:\n\n1."
            )

            output = model(
                conversations,
                temperature=temperature,
                top_p=0.95,
                max_tokens=512,
            )

        return output["choices"][0]["text"] if not instruct else \
            output["choices"][0]["message"]["content"]
    # Remote / server backend: talk to an OpenAI-compatible endpoint (a hosted
    # server, a vLLM server, or a llama.cpp server) using the official `openai`
    # client. The client handles retries + exponential backoff + timeouts for us,
    # so we no longer hand-roll a request loop. The request fields and the way we
    # read the response are kept identical to the previous implementation, so the
    # output contract is unchanged.
    import os

    from openai import OpenAI

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
        import httpx

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
                max_tokens=512,
            )
            return response.choices[0].message.content.replace("</s>", "")

        prompt = prompt + "Assistant: Sure, here is the requested answer:\n\n1."
        if start_flag is not None:
            prompt += start_flag

        response = client.completions.create(
            model=request_model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=512,
        )
        return response.choices[0].text
    except Exception as exc:
        # Surface a clear error instead of hanging or leaking a raw client error.
        raise RuntimeError(
            f"LLM request to {base_url} failed after {max_retries} retries"
        ) from exc