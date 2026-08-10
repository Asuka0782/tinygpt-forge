# Roadmap and scope control

## Verified baseline

- Shared-parameter manual and SDPA attention.
- RMSNorm, RoPE, configurable MHA/MQA/GQA, SwiGLU, and pre-norm decoder blocks.
- Dynamic and Static KV Cache with full/incremental equivalence.
- Character tokenizer, pre-fit train/validation/test split, local training and generation.
- Pickle-free weights and complete training continuation state.
- CPU regression suite, CUDA correctness probes, and raw-sample benchmark harness.
- Experimental OpenAI-compatible client with offline transport tests.

## Next release candidates

1. Add BPE or Unigram tokenization behind an optional dependency and compare vocabulary size, OOV,
   throughput, compression, and round-trip behavior.
2. Measure FP32/BF16 and eager/`torch.compile` under a fixed shape grid. Compile remains off by
   default until repeated steady-state gains exceed compile cost for a documented workload.
3. Add gradient checkpointing and measure throughput/memory trade-offs.
4. Add an Accelerate/DDP entry point on Linux and verify single-process equivalence before
   multi-process scaling.
5. Add optional PEFT LoRA integration with no-op, gradient, save/reload, disable, and merge tests.
6. Freeze the source walkthrough only after public APIs stop moving.

## Experimental research track

- Static Cache compilation and native GQA backend behavior.
- Weight-only INT8/INT4 using torchao or another compatibility-checked backend.
- Speculative decoding with target-distribution equivalence and acceptance-rate evidence.
- Prefix caching and a small continuous-batching scheduler.
- Paged-cache fragmentation simulator before any runtime claim.
- OpenAI-compatible local server only after input limits, cancellation, timeout, and concurrency
  semantics are designed and tested.

## Explicit non-goals for the first release

- SOTA language quality or training throughput.
- Production multi-tenant serving.
- Multi-node FSDP performance claims from a single Windows laptop.
- Bundling third-party model weights, private corpora, or paid-service credentials.
- Listing a feature as supported because another project supports it.

