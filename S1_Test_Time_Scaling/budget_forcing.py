"""
budget_forcing.py — the heart of s1: Simple Test-Time Scaling (Muennighoff et al., 2025).

"Budget forcing" is a decoding-time intervention that controls *how long* a
reasoning model thinks, with no retraining:

  * LENGTHEN  — when the model tries to stop thinking (emits the end-of-thinking
                delimiter), we suppress it and append "Wait" so it keeps going.
                More thinking -> more test-time compute -> (often) better answers.

  * SHORTEN   — if the model exceeds a thinking-token budget, we force the
                end-of-thinking delimiter to cut reasoning off and make it answer.

That is the entire idea. Everything below is plumbing around those two moves.

This works with any "reasoning" chat model that wraps its chain-of-thought in a
delimiter. We default to DeepSeek-R1-Distill (which uses <think> ... </think>).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Result:
    question: str
    thinking: str           # the (possibly forced) chain-of-thought
    answer: str             # text after the end-of-thinking delimiter
    thinking_tokens: int    # how many tokens of thinking were generated
    waits_injected: int     # how many times we forced "Wait" (the scaling knob)
    capped: bool            # True if we hit the max budget and cut thinking off
    full_text: str = field(repr=False)  # everything, for debugging


def pick_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


class BudgetForcer:
    """Wraps a HF causal LM and applies budget forcing during generation."""

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        think_open: str = "<think>",
        think_close: str = "</think>",
    ):
        auto_device, auto_dtype = pick_device_and_dtype()
        self.device = device or auto_device
        self.dtype = dtype or auto_dtype
        self.think_open = think_open
        self.think_close = think_close

        print(f"[s1] loading {model_name} on {self.device} ({self.dtype}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    # --- prompt construction -------------------------------------------------

    def _build_prefix(self, question: str, system_prompt: str | None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Make sure we have actually *entered* the thinking block. Some chat
        # templates already append "<think>"; if not, we add it ourselves.
        if self.think_open not in text.split("assistant")[-1]:
            text += f"{self.think_open}\n"
        return text

    # --- one chunk of greedy/sampled generation ------------------------------

    @torch.no_grad()
    def _generate_segment(self, text: str, max_new_tokens: int, sample, temperature, top_p):
        ids = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        out = self.model.generate(
            ids,
            max_new_tokens=max(1, max_new_tokens),
            do_sample=sample,
            temperature=temperature if sample else None,
            top_p=top_p if sample else None,
            stop_strings=[self.think_close],   # stop as soon as it tries to end thinking
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_ids = out[0][ids.shape[1]:]
        seg = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return seg, int(new_ids.shape[0])

    # --- the public entry point: generate with budget forcing ----------------

    def generate(
        self,
        question: str,
        max_thinking_tokens: int = 4096,   # SHORTEN knob: hard cap on thinking
        num_ignore: int = 0,               # LENGTHEN knob: how many times to force "Wait"
        wait_phrase: str = "Wait",
        answer_max_tokens: int = 512,
        system_prompt: str | None = None,
        sample: bool = False,              # greedy by default -> reproducible demos
        temperature: float = 0.6,
        top_p: float = 0.95,
        verbose: bool = False,
    ) -> Result:
        prefix = self._build_prefix(question, system_prompt)
        running = prefix
        thinking = ""
        budget_left = max_thinking_tokens
        waits_left = num_ignore
        waits_done = 0
        capped = False

        while True:
            seg, n_new = self._generate_segment(
                running, budget_left, sample, temperature, top_p
            )
            budget_left -= n_new

            if self.think_close in seg:
                # The model wants to stop thinking.
                before = seg.split(self.think_close)[0]
                thinking += before
                if waits_left > 0 and budget_left > 0:
                    # LENGTHEN: drop the delimiter, inject "Wait", keep thinking.
                    waits_left -= 1
                    waits_done += 1
                    running += before + " " + wait_phrase
                    thinking += " " + wait_phrase
                    if verbose:
                        print(f"\n[s1] forcing more thought (wait #{waits_done}) ...")
                    continue
                # Accept the end of thinking.
                running += before + self.think_close
                break

            # No delimiter in this segment.
            thinking += seg
            running += seg
            if budget_left <= 0:
                # SHORTEN: ran out of thinking budget -> force the model to answer.
                capped = True
                running += f"\n{self.think_close}\n"
                if verbose:
                    print("\n[s1] thinking budget exhausted -> forcing answer ...")
                break
            # Otherwise the model emitted EOS while still "thinking"; close it out.
            running += f"\n{self.think_close}\n"
            break

        # Now generate the final answer (after </think>), stopping at EOS.
        answer, _ = self._generate_segment(
            running, answer_max_tokens, sample, temperature, top_p
        )
        # answer segment may still trip the </think> stop string; strip it.
        answer = answer.replace(self.think_close, "").strip()
        running += answer

        thinking_tokens = max_thinking_tokens - budget_left
        return Result(
            question=question,
            thinking=thinking.strip(),
            answer=answer,
            thinking_tokens=thinking_tokens,
            waits_injected=waits_done,
            capped=capped,
            full_text=running,
        )
