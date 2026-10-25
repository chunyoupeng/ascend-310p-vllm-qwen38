#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import gc
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    UnquantizedEmbeddingMethod,
)

from vllm_ascend.ops.vocab_parallel_embedding import AscendParallelLMHead, AscendVocabParallelEmbedding
from vllm_ascend.utils import maybe_trans_nz


class AscendUnquantizedEmbeddingMethod310(UnquantizedEmbeddingMethod):
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if isinstance(layer, AscendParallelLMHead):
            layer.weight.data = maybe_trans_nz(layer.weight.data)
            layer.weight_nz = layer.weight
        elif hasattr(layer, "weight") and layer.weight.dtype == torch.float16 and layer.weight.shape[-1] % 2 == 0:
            w = layer.weight.data
            scale = (w.abs().amax(dim=-1, keepdim=True).to(torch.float32) / 127.0).clamp_(min=1e-8)
            w_int8 = torch.clamp(torch.round(w.to(torch.float32) / scale), -127, 127).to(torch.int8).contiguous()
            layer.weight_scale = torch.nn.Parameter(scale.to(torch.float16).contiguous(), requires_grad=False)
            # Store int8 bytes viewed as float16 [vocab, dim // 2] so aclnnEmbedding works in Eager and NPUGraph
            layer.weight = torch.nn.Parameter(w_int8.view(torch.float16), requires_grad=False)
            del w, scale, w_int8
            gc.collect()
            torch.npu.empty_cache()

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(layer, "weight_scale"):
            raw = F.embedding(x, layer.weight)
            q = raw.view(torch.int8).view(*x.shape, layer.embedding_dim).to(layer.weight_scale.dtype)
            s = F.embedding(x, layer.weight_scale)
            return q * s
        return F.embedding(x, layer.weight)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not hasattr(layer, "weight_nz"):
            layer.weight_nz = maybe_trans_nz(layer.weight)
        return F.linear(x, layer.weight_nz, bias)


class AscendVocabParallelEmbedding310(AscendVocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            num_embeddings, embedding_dim, params_dtype, org_num_embeddings, padding_size, quant_config, prefix
        )
        if quant_config is None or type(self.quant_method) is UnquantizedEmbeddingMethod:
            self.quant_method = AscendUnquantizedEmbeddingMethod310()


class AscendParallelLMHead310(AscendParallelLMHead):
    """
    Register ParallelLMHead as a custom op for Atlas 310p.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            num_embeddings, embedding_dim, bias, params_dtype, org_num_embeddings, padding_size, quant_config, prefix
        )

        if quant_config is None or type(self.quant_method) is UnquantizedEmbeddingMethod:
            self.quant_method = AscendUnquantizedEmbeddingMethod310()
