# Source, license, adoption, implementation, and evidence matrix

This project borrows ideas and interface lessons, not uncredited source. The current TinyGPT Forge
Python implementation was rewritten for this repository. If direct code adaptation is added later,
the affected file must preserve all required copyright and license notices.

| Feature or lesson | Primary source | License / paper | Decision | Current implementation | Evidence |
|---|---|---|---|---|---|
| Minimal GPT teaching structure | [minGPT](https://github.com/karpathy/minGPT) | MIT | Design reference | Rewritten, modular model | CPU shape/causality tests |
| Small training baseline | [nanoGPT](https://github.com/karpathy/nanoGPT) | MIT | Historical reference | Different training/checkpoint design | Integration test |
| End-to-end small-model workflow | [nanochat](https://github.com/karpathy/nanochat) | MIT | Organization reference | Partial local workflow | Train/generate CLI |
| RoPE/RMSNorm/SwiGLU/MQA | [llama2.c](https://github.com/karpathy/llama2.c) and papers | MIT + papers | Adopt core semantics | Yes | property/shape tests |
| Configured recipes and LoRA | [LitGPT](https://github.com/Lightning-AI/litgpt) | Apache-2.0 | Reference; LoRA deferred | No stable LoRA yet | N/A |
| HF interoperability | [Transformers](https://github.com/huggingface/transformers) | Apache-2.0 | Optional future bridge | No | N/A |
| Training recipes | [torchtune](https://github.com/meta-pytorch/torchtune) | BSD-3-Clause | Historical reference; maintenance wound down | No dependency | N/A |
| SDPA and GQA semantics | [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention) | PyTorch license | Adopt with fallback | Yes | forward/backward + dispatch record |
| FSDP | [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) | PyTorch license | Defer until Linux multi-GPU evidence | No | N/A |
| FlashAttention | [official repository](https://github.com/Dao-AILab/flash-attention) | BSD-3-Clause | Optional only | Unavailable in tested wheel | forced-backend failure probe |
| Paged KV and continuous batching | [vLLM](https://github.com/vllm-project/vllm) and PagedAttention paper | Apache-2.0 + paper | Experimental future work | No | N/A |
| Prefix/radix caching | [SGLang](https://github.com/sgl-project/sglang) | Apache-2.0 | Future systems reference | No | N/A |
| Local quantized inference | [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | Interoperability reference | No | N/A |
| BPE/Unigram tokenizer | [HF Tokenizers](https://github.com/huggingface/tokenizers) | Apache-2.0 | Optional next release | Character baseline only | tokenizer tests |
| SentencePiece | [SentencePiece](https://github.com/google/sentencepiece) | Apache-2.0 | Alternative under evaluation | No | N/A |
| Safe tensor serialization | [safetensors](https://github.com/huggingface/safetensors) | Apache-2.0 | Adopt | Model, optimizer, RNG tensors | exact reload/tamper tests |
| LoRA | [PEFT](https://github.com/huggingface/peft) and [LoRA paper](https://arxiv.org/abs/2106.09685) | Apache-2.0 + paper | Optional future bridge | No stable feature | N/A |
| QLoRA/4-bit | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) and QLoRA paper | MIT + paper | Deferred compatibility experiment | No | N/A |
| PyTorch-native quantization | [torchao](https://github.com/pytorch/ao) | BSD-3-Clause | Preferred first evaluation | No | N/A |
| Distributed launcher | [Accelerate](https://github.com/huggingface/accelerate) | Apache-2.0 | Optional future entry | No | N/A |
| Large-scale training organization | [TorchTitan](https://github.com/pytorch/torchtitan) | BSD-3-Clause | Manifest/checkpoint reference | Small safe manifest implemented | resume tests |
| Speculative decoding | [Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf) | ICML paper | Future prototype | No | N/A |
| Optional compatible API | OpenAI-style chat-completions schema | Interface convention | Experimental, local-first | Client only | offline transport/security tests |
| Reproducible benchmark figure | [Matplotlib](https://matplotlib.org/) and [NumPy](https://numpy.org/) | PSF-based + BSD-3-Clause | Optional plotting extra | JSON-driven script | regenerated PNG/PDF |

Core papers: [RoPE](https://arxiv.org/abs/2104.09864),
[RMSNorm](https://papers.neurips.cc/paper_files/paper/2019/file/1e8a19426224ca89e83cef47f1e7f53b-Paper.pdf),
[SwiGLU](https://arxiv.org/abs/2002.05202), and [GQA](https://arxiv.org/abs/2305.13245).

## Repository license status

The project owner selected Apache-2.0 on 2026-08-10. The canonical license text is included in the
repository root and the package metadata uses the SPDX identifier `Apache-2.0`. This project-level
license does not remove or replace any third-party license, notice, data, or paper obligation.
