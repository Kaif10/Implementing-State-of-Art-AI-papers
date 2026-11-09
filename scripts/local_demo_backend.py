"""Small real Hugging Face backend for demonstrations of API-oriented examples.

Implements the client methods used by this collection. Every generation comes
from the local checkpoint; no scripted model responses are supplied. Retrieval
embeddings use normalized mean token embeddings from the same checkpoint.
"""
import json
import os
from types import SimpleNamespace as NS

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from demo_quantization import quantize_cpu_linears


class LocalDemoClient:
    def __init__(self, model=None):
        model = model or os.environ.get("DEMO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        print(f"LOCAL BACKEND: {model} / CPU / real generation", flush=True)
        self.name = model
        torch.manual_seed(42)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        quantized = os.environ.get("DEMO_QUANTIZE") == "1"
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.bfloat16 if quantized else torch.float32).eval()
        if os.environ.get("DEMO_QUANTIZE") == "1":
            print("CPU inference: dynamic int8 linear layers", flush=True)
            self.model = quantize_cpu_linears(self.model)
        self.calls = 0
        self.chat = NS(completions=NS(create=self.chat_create))
        self.responses = NS(create=self.responses_create, parse=self.responses_parse)
        self.embeddings = NS(create=self.embed)

    def generate(self, messages, max_tokens=192, temperature=0, logprobs=False):
        self.calls += 1
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        n = inputs["input_ids"].shape[1]
        print(f"MODEL CALL {self.calls} / {n} input tokens", flush=True)
        kwargs = dict(max_new_tokens=min(max_tokens, 256), do_sample=temperature > 0,
                      pad_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=True,
                      output_scores=logprobs)
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=.9)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        ids = output.sequences[0, n:]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        print(f"MODEL CALL {self.calls} / {len(ids)} generated tokens", flush=True)
        tokens = None
        if logprobs:
            tokens = []
            for token, scores in zip(ids, output.scores):
                if int(token) in self.tokenizer.all_special_ids:
                    continue
                piece = self.tokenizer.decode([token], skip_special_tokens=True)
                prob = torch.log_softmax(scores.float(), -1)[0, token].item()
                tokens.append(NS(token=piece, bytes=list(piece.encode("utf-8")), logprob=prob))
        return text, tokens

    def chat_create(self, *, messages, max_tokens=192, max_completion_tokens=None,
                    temperature=0, logprobs=False, response_format=None, **kwargs):
        if response_format:
            messages = list(messages) + [{"role": "user", "content": "Return only valid JSON. Schema: " + json.dumps(response_format)}]
        text, tokens = self.generate(messages, max_completion_tokens or max_tokens, temperature, logprobs)
        return NS(choices=[NS(message=NS(content=text), logprobs=NS(content=tokens) if tokens else None)])

    def responses_create(self, *, input, max_output_tokens=192, temperature=0, text=None, **kwargs):
        messages = [{"role": "user", "content": input}] if isinstance(input, str) else list(input)
        if text:
            messages.append({"role": "user", "content": "Return ONLY the requested JSON object, without markdown or explanations."})
        generated, _ = self.generate(messages, max_output_tokens, temperature)
        return NS(output_text=generated)

    def responses_parse(self, *, input, text_format, temperature=0, **kwargs):
        schema = text_format.model_json_schema()
        messages = list(input) + [{"role": "user", "content":
            "Reply ONLY with a JSON object matching this schema. Use the exact enum values. " + json.dumps(schema)}]
        last_error = None
        for attempt in range(3):
            text, _ = self.generate(messages, 256, temperature)
            start, end = text.find("{"), text.rfind("}")
            try:
                parsed = text_format.model_validate_json(text[start:end+1] if start >= 0 else text)
                return NS(output_parsed=parsed)
            except ValueError as error:
                last_error = error
                messages.extend([{"role": "assistant", "content": text},
                                 {"role": "user", "content": "Correct the JSON. Output a complete object with only the required fields and valid enum values."}])
        raise ValueError("Local model did not produce schema-valid output after three attempts") from last_error

    def embed(self, *, input, dimensions=None, **kwargs):
        encoded = self.tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        mask = encoded["attention_mask"].unsqueeze(-1)
        with torch.inference_mode():
            vectors = (self.model.get_input_embeddings()(encoded["input_ids"]) * mask).sum(1) / mask.sum(1)
            if dimensions:
                vectors = torch.nn.functional.pad(vectors, (0, max(0, dimensions-vectors.shape[1])))[:, :dimensions]
            vectors = torch.nn.functional.normalize(vectors, dim=1)
        print(f"LOCAL EMBEDDINGS: {len(input)} texts / mean token vectors", flush=True)
        return NS(data=[NS(embedding=row.tolist()) for row in vectors])


class LocalDemoLLM:
    """Interface used by Reflexion's actor, judge and reflector."""
    def __init__(self, client):
        self.client = client

    def call(self, system, user, *, temperature=0, max_tokens=192, json_object=False):
        return self.client.responses_create(
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature, max_output_tokens=max_tokens,
            text={"format": {"type": "json_object"}} if json_object else None).output_text
