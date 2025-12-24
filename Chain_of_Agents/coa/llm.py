import time
import tiktoken
from openai import OpenAI


def get_encoder(model_name: str):
    # If tiktoken doesn't know the model, fall back to a common encoding.
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text or ""))


def generate(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int,
    retries: int = 0,
    retry_sleep_s: float = 1.0,
) -> str:
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            return (resp.output_text or "").strip()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_sleep_s)
    raise last_err
