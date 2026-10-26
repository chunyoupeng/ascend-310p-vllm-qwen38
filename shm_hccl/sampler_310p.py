import torch
import vllm.envs as envs
from vllm._custom_ops import apply_repetition_penalties
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.utils import get_token_bin_counts_and_mask
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

import vllm_ascend.sample.rejection_sampler as _rs_mod
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
    if _DEFAULT_NPU_GEN_310P is None or _DEFAULT_NPU_GEN_310P.device != device:
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


def generate_uniform_probs_310p(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """TP-synchronized FP32 uniform random numbers on NPU (avoids AICPU FP64 & TP RNG desync)."""
    default_gen = _get_default_npu_generator_310p(device)
    uniform_probs = torch.empty((num_tokens,), dtype=torch.float32, device=device)
    uniform_probs.uniform_(1e-6, 1.0, generator=default_gen)
    if generators:
        start_idx = 0
        for req_idx, n in enumerate(num_draft_tokens):
            if n == 0:
                continue
            end_idx = start_idx + n
            gen = generators.get(req_idx)
            if gen is not None:
                npu_gen = _get_npu_generator_310p(req_idx, gen, device)
                uniform_probs[start_idx:end_idx].uniform_(1e-6, 1.0, generator=npu_gen)
            start_idx = end_idx
    return uniform_probs


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


def _prepare_sampling_rejection_310p(self, top_k):
    self.top_k = top_k if top_k is not None else None
    if hasattr(self, "sampler") and hasattr(self.sampler, "prepare_sampling"):
        self.sampler.prepare_sampling(top_k)


def _sample_recovered_tokens_310p(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
    use_block_verify: bool = False,
    target_indices: torch.Tensor | None = None,
    global_vocab_size: int | None = None,
    enable_reduce_sampling: bool = False,
) -> torch.Tensor:
    num_tokens, vocab_size = target_probs.shape
    q = torch.empty((num_tokens, vocab_size), dtype=torch.float32, device=device)
    default_gen = _get_default_npu_generator_310p(device)
    q.exponential_(generator=default_gen)
    if sampling_metadata.generators:
        start_idx = 0
        for req_idx, n in enumerate(num_draft_tokens):
            if n == 0:
                continue
            end_idx = start_idx + n
            gen = sampling_metadata.generators.get(req_idx)
            if gen is not None:
                npu_gen = _get_npu_generator_310p(req_idx, gen, device)
                q[start_idx:end_idx].exponential_(generator=npu_gen)
            start_idx = end_idx

    recovered_token_ids = torch.empty_like(draft_token_ids)
    _sample_recovered_tokens_pytorch_310p(
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        IS_NGRAM=draft_probs is None,
        target_indices=target_indices,
        enable_reduce_sampling=enable_reduce_sampling,
    )
    return recovered_token_ids


def _sample_recovered_tokens_pytorch_310p(
    output_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    draft_probs,
    target_probs,
    q,
    vocab_size,
    IS_NGRAM=False,
    target_indices=None,
    enable_reduce_sampling=False,
):
    device = output_token_ids.device
    num_tokens = output_token_ids.shape[0]
    if num_tokens == 0:
        return

    if q.shape[0] == num_tokens:
        q_values = q
    elif q.shape[0] == 1:
        q_values = q.expand(num_tokens, vocab_size)
    else:
        cu_start = torch.nn.functional.pad(cu_num_draft_tokens[:-1], (1, 0), value=0)
        token_indices = torch.arange(num_tokens, device=device)[:, None]
        in_range_mask = (token_indices >= cu_start[None, :]) & (token_indices < cu_num_draft_tokens[None, :])
        token_to_batch = torch.argmax(in_range_mask.int(), dim=1)
        q_values = q[token_to_batch]

    if enable_reduce_sampling:
        if IS_NGRAM:
            prob = target_probs
        else:
            flat_indices = target_indices.flatten()
            token_offsets = torch.arange(num_tokens, device=device)[:, None] * draft_probs.shape[1]
            flat_token_offsets = token_offsets.expand_as(target_indices).flatten()
            draft_probs_flat = draft_probs.flatten()
            valid_mask = flat_indices < draft_probs.shape[1]
            flat_draft_probs_at_indices = torch.where(
                valid_mask, draft_probs_flat[flat_token_offsets + flat_indices], 0.0
            )
            draft_probs_at_indices = flat_draft_probs_at_indices.view(num_tokens, vocab_size)
            prob = torch.clamp_min(target_probs - draft_probs_at_indices, 0.0)
    else:
        if IS_NGRAM:
            prob = target_probs
        else:
            prob = torch.clamp_min(target_probs - draft_probs, 0.0)

    q_values_safe = torch.clamp_min(q_values, 1e-10)
    prob_over_q = prob / q_values_safe

    if enable_reduce_sampling:
        indices = torch.argmax(prob_over_q, dim=1)
        output_token_ids[:] = target_indices.gather(1, indices.unsqueeze(1)).squeeze(1)
    else:
        output_token_ids[:] = torch.argmax(prob_over_q, dim=1)


def _rejection_random_sample_pytorch_310p(
    output_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    draft_probs,
    target_probs,
    bonus_token_ids,
    recovered_token_ids,
    uniform_probs,
    is_greedy,
    max_spec_len,
    vocab_size,
    IS_NGRAM=False,
    target_indices=None,
    enable_reduce_sampling=False,
    ENTROPY_VERIFY=False,
    POSTERIOR_THRESHOLD=0.95,
    POSTERIOR_ALPHA=0.4,
    EPSILON=1e-10,
    ori_target_probs=None,
):
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    cu_start = torch.nn.functional.pad(cu_num_draft_tokens[:-1], (1, 0), value=0)
    num_draft_per_batch = cu_num_draft_tokens - cu_start

    max_draft_len = max_spec_len
    pos_indices = torch.arange(max_draft_len, device=device)[None, :]

    valid_mask = pos_indices < num_draft_per_batch[:, None]
    global_token_indices = (cu_start[:, None] + pos_indices).clamp(0, draft_token_ids.shape[0] - 1)
    draft_tokens = draft_token_ids[global_token_indices]
    placeholder_mask = draft_tokens == PLACEHOLDER_TOKEN_ID
    safe_draft_tokens = draft_tokens.masked_fill(placeholder_mask, 0)

    if enable_reduce_sampling:
        flat_global_indices = global_token_indices.flatten()
        flat_draft_tokens = draft_tokens.flatten().unsqueeze(1)
        flat_target_indices = target_indices[flat_global_indices]
        flat_target_probs = target_probs[flat_global_indices]
        is_in_candidates = flat_target_indices == flat_draft_tokens
        target_token_probs = (flat_target_probs * is_in_candidates).sum(dim=1).view(batch_size, max_draft_len)
    else:
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = safe_draft_tokens.flatten()
        target_token_probs = target_probs[flat_indices, flat_draft_tokens].view(batch_size, max_draft_len)

    uniform_token_probs = uniform_probs[global_token_indices]
    recovered_tokens = recovered_token_ids[global_token_indices]

    if IS_NGRAM:
        acceptance_condition = (target_token_probs >= uniform_token_probs) | (draft_tokens == recovered_tokens)
    else:
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = safe_draft_tokens.flatten()
        draft_token_probs = draft_probs[flat_indices, flat_draft_tokens].view(batch_size, max_draft_len)
        acceptance_condition = (draft_token_probs > 0.0) & (
            target_token_probs / draft_token_probs >= uniform_token_probs
        )
    acceptance_condition = acceptance_condition & (~placeholder_mask)

    first_rejection = (~acceptance_condition) & valid_mask
    default_pos = torch.full((batch_size, 1), max_draft_len, dtype=torch.long, device=device)
    first_reject_pos = torch.where(
        first_rejection.any(dim=1, keepdim=True),
        first_rejection.int().argmax(dim=1, keepdim=True),
        default_pos,
    )
    pos_mask = pos_indices >= first_reject_pos
    should_skip = pos_mask & valid_mask

    final_acceptance = acceptance_condition & (~should_skip)
    non_greedy_mask = ~is_greedy
    update_mask = non_greedy_mask[:, None] & valid_mask & (~should_skip)

    first_reject_mask = (pos_indices == first_reject_pos) & valid_mask & non_greedy_mask[:, None]
    final_update_mask = update_mask | first_reject_mask
    final_tokens = torch.where(
        first_reject_mask,
        recovered_tokens,
        torch.where(final_acceptance, draft_tokens, output_token_ids[:, :max_draft_len]),
    )

    output_token_ids[:, :max_draft_len] = torch.where(
        final_update_mask, final_tokens, output_token_ids[:, :max_draft_len]
    )

    no_rejection = first_reject_pos.squeeze(1) >= num_draft_per_batch
    should_add_bonus = non_greedy_mask & no_rejection
    bonus_positions = num_draft_per_batch
    seq_len = output_token_ids.shape[1]
    all_positions = torch.arange(seq_len, device=device)[None, :]
    valid_bonus_pos = bonus_positions < (max_spec_len + 1)
    final_bonus_mask = should_add_bonus & valid_bonus_pos
    bonus_pos_mask = (all_positions == bonus_positions[:, None]) & final_bonus_mask[:, None]
    bonus_values_expanded = bonus_token_ids.view(-1, 1).expand(-1, seq_len)
    output_token_ids[:] = torch.where(bonus_pos_mask, bonus_values_expanded, output_token_ids)


def _expand_pytorch_310p(
    output_ptr,
    input_ptr,
    cu_num_tokens_ptr,
    replace_from,
    replace_to,
    MAX_NUM_TOKENS,
):
    device = cu_num_tokens_ptr.device
    batch_size = input_ptr.shape[0]
    num_tokens = output_ptr.shape[0]
    if batch_size == 0 or num_tokens == 0:
        return
    if batch_size == 1:
        val = torch.where(input_ptr == replace_from, replace_to, input_ptr)
        output_ptr[:] = val[0]
        return
    cu_start = torch.nn.functional.pad(cu_num_tokens_ptr[:-1], (1, 0), value=0)
    token_indices = torch.arange(num_tokens, device=device)[:, None]
    in_range = (token_indices >= cu_start[None, :]) & (token_indices < cu_num_tokens_ptr[None, :])
    replaced_input = torch.where(input_ptr == replace_from, replace_to, input_ptr).float()
    token_values = torch.einsum("tb,b->t", in_range.float(), replaced_input)
    needs_update = in_range.any(dim=1)
    output_ptr[:] = torch.where(needs_update, token_values, output_ptr)


_orig_rejection_forward_310p = AscendRejectionSampler.forward


def _ascend_rejection_sampler_forward_310p(
    self,
    metadata: SpecDecodeMetadata,
    draft_probs: torch.Tensor | None,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
):
    _rs_mod.sample_recovered_tokens = _sample_recovered_tokens_310p
    return _orig_rejection_forward_310p(self, metadata, draft_probs, logits, sampling_metadata)


AscendSampler.apply_penalties = _ascend_sampler_apply_penalties_310p
AscendRejectionSampler.apply_penalties = _ascend_rejection_sampler_apply_penalties_310p
AscendRejectionSampler.prepare_sampling = _prepare_sampling_rejection_310p
AscendRejectionSampler.forward = _ascend_rejection_sampler_forward_310p
_rs_mod.generate_uniform_probs = generate_uniform_probs_310p
_rs_mod.sample_recovered_tokens = _sample_recovered_tokens_310p
_rs_mod.sample_recovered_tokens_pytorch = _sample_recovered_tokens_pytorch_310p
_rs_mod.rejection_random_sample_pytorch = _rejection_random_sample_pytorch_310p
_rs_mod.expand_pytorch = _expand_pytorch_310p


class AscendTopKTopPSampler310(AscendTopKTopPSampler):
    def forward_native(self, logits, generators, k, p):
        if envs.VLLM_BATCH_INVARIANT:
            return super().forward_native(logits, generators, k, p)
        if get_ascend_config().enable_reduce_sample:
            top_k = self.top_k if self.top_k is not None else 64
            cand_logits, cand_idx = self.apply_top_k_top_p(logits, k, p, top_k)
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
