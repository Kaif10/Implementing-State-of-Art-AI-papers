"""
Speculative Decoding — faithful implementation of the core algorithm from:

    Leviathan, Kalman & Matias, "Fast Inference from Transformers via
    Speculative Decoding", ICML 2023.  https://arxiv.org/abs/2211.17192
    (Concurrently: Chen et al., "Accelerating LLM Decoding with Speculative
    Sampling", https://arxiv.org/abs/2302.01318)

Idea
----
A small, cheap *draft* model proposes K tokens autoregressively. A large,
accurate *target* model then scores all K proposals in a SINGLE forward pass.
A rejection-sampling acceptance test decides how many proposals to keep. The
key theoretical result (Theorem 1 / Appendix in the paper) is that the tokens
produced are distributed EXACTLY as if they had been sampled from the target
model alone — speculative decoding is a lossless accelerator, not an
approximation.

Per round, the target model emits at least 1 and up to K+1 tokens, while only
ever doing 1 forward pass. The expected number of accepted tokens is the source
of the wall-clock speedup.

This file implements the entire core logic from scratch: the draft loop, the
parallel target scoring, the accept/reject rule, the residual (p - q)+
resampling on rejection, and the bonus token when all K are accepted.
"""

import torch
import torch.nn.functional as F
from transformers import DynamicCache


# --------------------------------------------------------------------------- #
# Probability helpers                                                         #
# --------------------------------------------------------------------------- #
def logits_to_probs(logits, temperature, top_k=0):
    """Convert a [.., vocab] logits row to a probability distribution.

    Matches the sampling distribution used everywhere so that the draft model's
    q(x) and the target model's p(x) are directly comparable in the acceptance
    test. temperature == 0 is treated as greedy (a one-hot at the argmax).
    """
    if temperature <= 0:
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        return probs
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return F.softmax(logits, dim=-1)


def sample(probs):
    """Draw one token id from a [1, vocab] probability row -> [1, 1]."""
    return torch.multinomial(probs, num_samples=1)


# --------------------------------------------------------------------------- #
# Baseline: ordinary autoregressive sampling from the target model alone      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def autoregressive_generate(model, input_ids, max_new_tokens,
                            temperature=1.0, top_k=0):
    """Standard one-token-at-a-time sampling. The thing we want to speed up.

    Returns (output_ids, stats). Used as the correctness/latency reference.
    """
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits[:, -1, :]
        probs = logits_to_probs(logits, temperature, top_k)
        next_token = sample(probs)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return input_ids, {"target_forwards": max_new_tokens}


# --------------------------------------------------------------------------- #
# Speculative decoding                                                        #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def speculative_generate(target_model, draft_model, input_ids, max_new_tokens,
                         K=4, temperature=1.0, top_k=0):
    """Generate with speculative decoding. Lossless w.r.t. sampling from the
    target model at the same temperature/top_k.

    Args:
        target_model: large, accurate model (defines the true distribution p).
        draft_model:  small, fast model (proposal distribution q).
        K: number of tokens drafted per round (the "speculation length").

    Returns (output_ids, stats) where stats reports acceptance rate and the
    number of target forward passes — the basis for the measured speedup.
    """
    prompt_len = input_ids.shape[1]
    proposed_total = 0          # draft tokens scored by the target
    accepted_total = 0          # of those, how many were accepted
    target_forwards = 0         # large-model forward passes (the cost we cut)

    while input_ids.shape[1] - prompt_len < max_new_tokens:
        # --- 1. DRAFT: small model proposes K tokens autoregressively ----- #
        draft_seq = input_ids
        draft_tokens, draft_probs = [], []
        for _ in range(K):
            logits = draft_model(draft_seq).logits[:, -1, :]
            q = logits_to_probs(logits, temperature, top_k)   # q(.) for this step
            tok = sample(q)
            draft_tokens.append(tok)
            draft_probs.append(q)
            draft_seq = torch.cat([draft_seq, tok], dim=-1)

        drafted = torch.cat(draft_tokens, dim=-1)             # [1, K]

        # --- 2. VERIFY: target scores all K proposals in ONE forward pass - #
        # Feeding the prompt + K drafted tokens yields, at each position, the
        # target's distribution for the *next* token. Position (prompt_len-1+i)
        # gives p(.) for drafted token i; the final position gives the bonus.
        full = torch.cat([input_ids, drafted], dim=-1)
        target_logits = target_model(full).logits            # [1, L+K, vocab]
        target_forwards += 1
        base = input_ids.shape[1] - 1                         # idx predicting tok 0

        # --- 3. ACCEPT / REJECT each drafted token, left to right --------- #
        n_accepted = 0
        for i in range(K):
            p = logits_to_probs(target_logits[:, base + i, :], temperature, top_k)
            q = draft_probs[i]
            tok = draft_tokens[i]
            p_x = p.gather(-1, tok)        # target prob of the proposed token
            q_x = q.gather(-1, tok)        # draft  prob of the proposed token

            # Accept with probability min(1, p(x)/q(x)).
            r = torch.rand_like(p_x)
            proposed_total += 1
            if (r < torch.clamp(p_x / q_x, max=1.0)).item():
                input_ids = torch.cat([input_ids, tok], dim=-1)
                n_accepted += 1
            else:
                # Reject: resample from the residual distribution (p - q)+,
                # renormalised. This correction is what makes the overall
                # output distribution exactly p (paper, Theorem 1).
                residual = torch.clamp(p - q, min=0.0)
                residual = residual / residual.sum(dim=-1, keepdim=True)
                corrected = sample(residual)
                input_ids = torch.cat([input_ids, corrected], dim=-1)
                break
        else:
            # All K accepted -> we get a free "bonus" token sampled directly
            # from the target's distribution at the final position.
            p_last = logits_to_probs(target_logits[:, base + K, :], temperature, top_k)
            bonus = sample(p_last)
            input_ids = torch.cat([input_ids, bonus], dim=-1)

        accepted_total += n_accepted

    # Trim any overshoot from the last round to exactly max_new_tokens.
    output_ids = input_ids[:, : prompt_len + max_new_tokens]
    stats = {
        "proposed": proposed_total,
        "accepted": accepted_total,
        "acceptance_rate": (accepted_total / proposed_total) if proposed_total else 0.0,
        "target_forwards": target_forwards,
        "K": K,
    }
    return output_ids, stats


# --------------------------------------------------------------------------- #
# KV-cached variants                                                          #
# --------------------------------------------------------------------------- #
# The functions above recompute the whole sequence on every forward pass, which
# is O(n^2) and makes the small draft model deceptively expensive. Real serving
# uses the model's built-in KV cache (use_cache=True / past_key_values): each
# step only attends over cached keys/values, so a forward is O(1) in the
# sequence length. These cached variants are what reproduce the paper's
# wall-clock speedups. The ONLY speculative-specific bookkeeping is cropping the
# cache back when drafted tokens are rejected (the cache machinery itself lives
# inside the model).

def _prime(model, prompt_ids):
    """Fill a fresh KV cache with prompt[:-1] and return (cache, anchor).

    Invariant maintained throughout: the cache holds every committed token
    EXCEPT the last one (`anchor`); the anchor is fed fresh each round so its
    next-token logits are produced inside the verification pass.
    """
    cache = DynamicCache()
    if prompt_ids.shape[1] > 1:
        model(prompt_ids[:, :-1], past_key_values=cache, use_cache=True)
    return cache, prompt_ids[:, -1:]


@torch.no_grad()
def autoregressive_generate_cached(model, input_ids, max_new_tokens,
                                   temperature=1.0, top_k=0):
    """KV-cached autoregressive baseline: one cheap forward per token."""
    cache, anchor = _prime(model, input_ids)
    generated = input_ids
    for _ in range(max_new_tokens):
        out = model(anchor, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        probs = logits_to_probs(out.logits[:, -1, :], temperature, top_k)
        anchor = sample(probs)
        generated = torch.cat([generated, anchor], dim=-1)
    return generated, {"target_forwards": max_new_tokens}


@torch.no_grad()
def speculative_generate_cached(target_model, draft_model, input_ids,
                                max_new_tokens, K=4, temperature=1.0, top_k=0):
    """KV-cached speculative decoding. Lossless w.r.t. target sampling.

    One target forward per round (over the anchor + K drafted tokens); the draft
    runs K+1 single-token forwards. On rejection (or after the bonus token) both
    caches are cropped back to the committed length so no rejected key/value
    state lingers.
    """
    prompt_len = input_ids.shape[1]
    committed_len = prompt_len               # tokens decided so far
    proposed_total = accepted_total = target_forwards = 0

    t_cache, anchor = _prime(target_model, input_ids)
    d_cache, _ = _prime(draft_model, input_ids)
    generated = input_ids

    while committed_len - prompt_len < max_new_tokens:
        # --- 1. DRAFT: feed anchor, then each sampled token, into the draft -- #
        feed = anchor
        draft_tokens, draft_probs = [], []
        for _ in range(K):
            out = draft_model(feed, past_key_values=d_cache, use_cache=True)
            d_cache = out.past_key_values
            q = logits_to_probs(out.logits[:, -1, :], temperature, top_k)
            feed = sample(q)
            draft_tokens.append(feed)
            draft_probs.append(q)
        # One more draft forward so the draft cache also holds the last drafted
        # token (keeps cropping uniform regardless of how many get accepted).
        d_cache = draft_model(feed, past_key_values=d_cache, use_cache=True).past_key_values
        drafted = torch.cat(draft_tokens, dim=-1)              # [1, K]

        # --- 2. VERIFY: ONE target forward over [anchor, d_0..d_{K-1}] ------ #
        verify_in = torch.cat([anchor, drafted], dim=-1)       # [1, K+1]
        out = target_model(verify_in, past_key_values=t_cache, use_cache=True)
        t_cache = out.past_key_values
        target_forwards += 1
        # logits[:, j] predicts the token after verify_in[:, j]:
        #   j = 0      -> p(d_0)        (uses the anchor's full context)
        #   j = 1..K-1 -> p(d_j)
        #   j = K      -> p(bonus)
        verify_logits = out.logits                             # [1, K+1, vocab]

        # --- 3. ACCEPT / REJECT, vectorised over all K proposals ----------- #
        # Same rule as the readable per-token version in speculative_generate,
        # but computed in one shot so the accept run length costs a SINGLE
        # GPU<->CPU sync per round instead of one per token (critical on MPS).
        P = logits_to_probs(verify_logits[:, :K, :], temperature, top_k)   # [1,K,V]
        Q = torch.stack(draft_probs, dim=1)                                # [1,K,V]
        tok_idx = drafted.unsqueeze(-1)                                    # [1,K,1]
        p_x = P.gather(-1, tok_idx).squeeze(-1)                            # [1,K]
        q_x = Q.gather(-1, tok_idx).squeeze(-1)                            # [1,K]
        r = torch.rand_like(p_x)
        accept = r < torch.clamp(p_x / q_x, max=1.0)                       # [1,K] bool
        # n_accepted = length of the leading run of accepts (cumprod zeroes out
        # everything after the first rejection); one .item() for the whole round.
        n_accepted = int(accept.long().cumprod(dim=-1).sum().item())
        # Tokens actually verified this round: the accepted ones plus the single
        # rejected one (if any). Proposals after the first rejection are
        # discarded unevaluated, so they don't enter the acceptance rate (this
        # matches the paper's per-token acceptance probability alpha).
        proposed_total += n_accepted + (0 if n_accepted == K else 1)
        accepted_total += n_accepted

        if n_accepted < K:
            # Rejected proposal n_accepted -> resample from residual (p - q)+.
            residual = torch.clamp(P[:, n_accepted, :] - Q[:, n_accepted, :], min=0.0)
            residual = residual / residual.sum(dim=-1, keepdim=True)
            extra = sample(residual)
        else:                    # all K accepted -> free bonus token
            extra = sample(logits_to_probs(verify_logits[:, K, :], temperature, top_k))

        # --- 4. COMMIT + crop caches back to the committed prefix ---------- #
        if n_accepted:
            generated = torch.cat([generated, drafted[:, :n_accepted]], dim=-1)
        generated = torch.cat([generated, extra], dim=-1)
        committed_len += n_accepted + 1
        # Caches grew to (committed_len_before + K) entries; keep only the
        # accepted ones. The new anchor (extra) is intentionally left out so it
        # is fed fresh next round.
        keep = committed_len - 1                 # = old committed_len + n_accepted
        t_cache.crop(keep)
        d_cache.crop(keep)
        anchor = extra

    output_ids = generated[:, : prompt_len + max_new_tokens]
    stats = {
        "proposed": proposed_total,
        "accepted": accepted_total,
        "acceptance_rate": (accepted_total / proposed_total) if proposed_total else 0.0,
        "target_forwards": target_forwards,
        "K": K,
    }
    return output_ids, stats
