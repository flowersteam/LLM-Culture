import time

import requests

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
    # llama.cpp server
    if instruct:
        url = access_url.rstrip("/") + "/v1/chat/completions"

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": 512,
        }

    else:
        url = access_url.rstrip("/") + "/v1/completions"

        prompt = prompt + "Assistant: Sure, here is the requested answer:\n\n1."

        if start_flag is not None:
            prompt += start_flag

        data = {
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": temperature,
        }

    if debug:
        print("POST:", url)
        print("DATA:", data)

    last_error = None
    delay = 1
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=data,
                verify=verify,
                timeout=timeout,
            )
        except requests.exceptions.RequestException as exc:
            # connection error / timeout / etc.
            last_error = exc
            if debug:
                print(f"Request failed (attempt {attempt}/{max_retries}): {exc}")
        else:
            if debug:
                print("Status:", response.status_code)
                print("Response:", response.text)

            if response.ok:
                result = response.json()

                if instruct:
                    return result["choices"][0]["message"]["content"].replace("</s>", "")
                else:
                    return result["choices"][0]["text"]

            last_error = RuntimeError(
                f"server returned HTTP {response.status_code}: {response.text[:200]}"
            )
            if debug:
                print(f"Server error {response.status_code} (attempt {attempt}/{max_retries}), trying again...")

        # exponential backoff between attempts (1s, 2s, 4s, 8s, capped at 16s)
        if attempt < max_retries:
            time.sleep(delay)
            delay = min(delay * 2, 16)

    raise RuntimeError(
        f"LLM request to {url} failed after {max_retries} attempts"
    ) from last_error