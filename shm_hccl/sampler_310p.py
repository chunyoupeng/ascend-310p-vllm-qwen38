from collections import Counter
import torch
import vllm.envs as envs
from vllm._custom_ops import apply_repetition_penalties
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.utils import get_token_bin_counts_and_mask
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
from vllm_ascend.sample.sampler import (
    DEFAULT_LOGPROBS_MODE,
    AscendSampler,
    AscendTopKTopPSampler,
)

_DEFAULT_NPU_GEN_310P: torch.Generator | None = None
_NPU_GENERATOR_CACHE_310P: dict[int, tuple[torch.Generator, int]] = {}
_PROMPT_META_CACHE_310P: dict[tuple[int, tuple[int, ...], int, int], tuple[torch.Tensor, bool, torch.Tensor | None]] = {}


def _get_default_npu_generator_310p(device: torch.device) -> torch.Generator:
    global _DEFAULT_NPU_GEN_310P
    if _DEFAULT_NPU_GEN_310P is None:
        _DEFAULT_NPU_GEN_310P = torch.Generator(device=device)
        _DEFAULT_NPU_GEN_310P.manual_seed(20260417)
    return _DEFAULT_NPU_GEN_310P


def _get_npu_generator_310p(i: int, generator: torch.Generator, device: torch.device) -> torch.Generator:
    cache_entry = _NPU_GENERATOR_CACHE_310P.get(i)
    if cache_entry is None or cache_entry[1] != id(generator):
        npu_generator = torch.Generator(device=device)
        try:
            npu_generator.manual_seed(generator.initial_seed())
        except Exception:
            npu_generator.manual_seed(20260417 + i)
        cache_entry = (npu_generator, id(generator))
        _NPU_GENERATOR_CACHE_310P[i] = cache_entry
    return cache_entry[0]


def fill_exponential_310p(
    q: torch.Tensor,
    generators: dict[int, torch.Generator],
    has_draft_mask: torch.Tensor | None = None,
) -> None:
    """Fill q with exponential values directly on NPU using synchronized NPU RNG."""
    default_gen = _get_default_npu_generator_310p(q.device)
    if not generators:
        q.exponential_(generator=default_gen)
        return
    if len(generators) != q.shape[0] or has_draft_mask is not None:
        q.exponential_(generator=default_gen)
    if has_draft_mask is not None:
        has_draft_mask_cpu = has_draft_mask.cpu()
        for i, generator in generators.items():
            npu_gen = _get_npu_generator_310p(i, generator, q.device)
            if bool(has_draft_mask_cpu[i]):
                q[i].exponential_(generator=npu_gen)
    else:
        for i, generator in generators.items():
            npu_gen = _get_npu_generator_310p(i, generator, q.device)
            q[i].exponential_(generator=npu_gen)


def _random_sample_310p(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """310P-specific random sampling with NPU exponential generation for q."""
    q = torch.empty_like(probs)
    fill_exponential_310p(q, generators)
    return probs.div_(q).argmax(dim=-1).view(-1)


def _to_local_shard_tokens_310p(tokens: torch.Tensor, vocab_size: int, start_idx: int) -> torch.Tensor:
    if start_idx == 0:
        in_shard = (tokens >= 0) & (tokens < vocab_size)
        return torch.where(in_shard, tokens, vocab_size)
    local_t = tokens - start_idx
    in_shard = (local_t >= 0) & (local_t < vocab_size)
    return torch.where(in_shard, local_t, vocab_size)


def _get_prompt_meta_310p(
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    vocab_size: int,
    start_idx: int,
) -> tuple[bool, torch.Tensor | None]:
    key = (id(prompt_token_ids), tuple(prompt_token_ids.shape), vocab_size, start_idx)
    cached = _PROMPT_META_CACHE_310P.get(key)
    if cached is not None and cached[0] is prompt_token_ids:
        return cached[1], cached[2]
    if len(_PROMPT_META_CACHE_310P) >= 8:
        _PROMPT_META_CACHE_310P.clear()
    has_explicit = bool(
        (repetition_penalties > 1.001).any()
        or (repetition_penalties < 0.999).any()
        or (presence_penalties != 0.0).any()
        or (frequency_penalties != 0.0).any()
    )
    if has_explicit:
        local_prompt = _to_local_shard_tokens_310p(prompt_token_ids, vocab_size, start_idx)
        _, prompt_mask = get_token_bin_counts_and_mask(local_prompt, vocab_size, prompt_token_ids.shape[0])
    else:
        prompt_mask = None
    _PROMPT_META_CACHE_310P[key] = (prompt_token_ids, has_explicit, prompt_mask)
    return has_explicit, prompt_mask


def _detect_tail_loop_penalties_310p(
    output_token_ids: list[list[int]],
    vocab_size: int,
    start_idx: int,
) -> list[tuple[int, int, float]]:
    """Return list of (row_idx, local_token_idx, penalty_to_subtract) ONLY when a repetition loop occurs."""
    end_idx = start_idx + vocab_size
    penalties: list[tuple[int, int, float]] = []
    for row_idx, seq in enumerate(output_token_ids):
        n = len(seq)
        if n < 6:
            continue
        row_pen: dict[int, float] = {}
        tail = seq[-96:] if n > 96 else seq
        tlen = len(tail)
        # 1. Exact periodic n-gram repeat at the tail (period k in 1..24)
        max_k = min(24, tlen // 2)
        for k in range(1, max_k + 1):
            if tail[-k:] == tail[-2 * k : -k]:
                triple = (tlen >= 3 * k) and (tail[-3 * k : -2 * k] == tail[-k:])
                next_tok = tail[-k]
                if start_idx <= next_tok < end_idx:
                    c = next_tok - start_idx
                    row_pen[c] = max(row_pen.get(c, 0.0), 12.0 if triple else 5.0)
                if triple:
                    for tok in set(tail[-k:]):
                        if start_idx <= tok < end_idx:
                            c = tok - start_idx
                            row_pen[c] = max(row_pen.get(c, 0.0), 8.0)
        # 2. High-frequency burst in the last 32 tokens (token appears >= 5 times)
        short_tail = tail[-32:]
        counts = Counter(short_tail)
        for tok, cnt in counts.items():
            if cnt >= 5 and start_idx <= tok < end_idx:
                c = tok - start_idx
                row_pen[c] = max(row_pen.get(c, 0.0), (cnt - 4) * 1.5)
        for c, val in row_pen.items():
            penalties.append((row_idx, c, val))
    return penalties


def _apply_penalties_sharded_310p(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
    repeat_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    num_seqs, vocab_size = logits.shape
    if num_seqs == 0:
        return logits

    if get_ascend_config().enable_reduce_sample:
        rank = get_tp_group().rank_in_group
        start_idx = rank * vocab_size
    else:
        start_idx = 0

    # Fast surgical loop-breaker on CPU tail (0 NPU overhead when no loop)
    if output_token_ids:
        loop_penalties = _detect_tail_loop_penalties_310p(output_token_ids, vocab_size, start_idx)
        if loop_penalties:
            rows = [p[0] for p in loop_penalties]
            cols = [p[1] for p in loop_penalties]
            vals = torch.tensor([p[2] for p in loop_penalties], dtype=logits.dtype, device=logits.device)
            logits[rows, cols] -= vals

    has_explicit_penalties, base_prompt_mask = _get_prompt_meta_310p(
        prompt_token_ids,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        vocab_size,
        start_idx,
    )
    if not has_explicit_penalties or base_prompt_mask is None:
        return logits

    if repeat_indices is not None:
        prompt_mask = base_prompt_mask[repeat_indices]
        presence_penalties = presence_penalties[repeat_indices]
        frequency_penalties = frequency_penalties[repeat_indices]
        repetition_penalties = repetition_penalties[repeat_indices]
    else:
        prompt_mask = base_prompt_mask

    if not output_token_ids or all(len(x) == 0 for x in output_token_ids):
        output_bin_counts = torch.zeros((num_seqs, vocab_size), dtype=torch.long, device=logits.device)
        output_mask = torch.zeros((num_seqs, vocab_size), dtype=torch.bool, device=logits.device)
    else:
        output_tokens_t = make_tensor_with_pad(
            output_token_ids,
            pad=-1,
            device="cpu",
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(),
        ).to(logits.device, non_blocking=True)
        local_output_t = _to_local_shard_tokens_310p(output_tokens_t, vocab_size, start_idx)
        output_bin_counts, output_mask = get_token_bin_counts_and_mask(
            local_output_t, vocab_size, num_seqs
        )

    apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


@staticmethod
def _ascend_sampler_apply_penalties_310p(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    if sampling_metadata.no_penalties:
        return logits
    assert sampling_metadata.prompt_token_ids is not None
    return _apply_penalties_sharded_310p(
        logits,
        sampling_metadata.prompt_token_ids,
        sampling_metadata.presence_penalties,
        sampling_metadata.frequency_penalties,
        sampling_metadata.repetition_penalties,
        output_token_ids,
        repeat_indices=None,
    )


@staticmethod
def _ascend_rejection_sampler_apply_penalties_310p(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    metadata: SpecDecodeMetadata,
    repeat_indices: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    if sampling_metadata.no_penalties:
        return logits
    assert sampling_metadata.prompt_token_ids is not None
    return _apply_penalties_sharded_310p(
        logits,
        sampling_metadata.prompt_token_ids,
        sampling_metadata.presence_penalties,
        sampling_metadata.frequency_penalties,
        sampling_metadata.repetition_penalties,
        output_token_ids,
        repeat_indices=repeat_indices,
    )


AscendSampler.apply_penalties = _ascend_sampler_apply_penalties_310p
AscendRejectionSampler.apply_penalties = _ascend_rejection_sampler_apply_penalties_310p


class AscendTopKTopPSampler310(AscendTopKTopPSampler):
    def forward_native(self, logits, generators, k, p):
        if envs.VLLM_BATCH_INVARIANT:
            return super().forward_native(logits, generators, k, p)
        if get_ascend_config().enable_reduce_sample:
            cand_logits, cand_idx = self.apply_top_k_top_p(logits, k, p, self.top_k)
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = cand_logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = cand_logits.log_softmax(dim=-1, dtype=torch.float32)

            probs = torch.softmax(cand_logits, dim=-1)
            pos = _random_sample_310p(probs, generators)  # [B]

            next_token = cand_idx.gather(dim=1, index=pos.unsqueeze(1)).squeeze(1)  # [B]
            return next_token, logits_to_return
        else:
            logits = self.apply_top_k_top_p(logits, k, p)
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

            probs = logits.softmax(dim=-1, dtype=torch.float32)
            return _random_sample_310p(probs, generators), logits_to_return


class AscendSampler310(AscendSampler):
    apply_penalties = _ascend_sampler_apply_penalties_310p

    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler310(logprobs_mode=logprobs_mode)
