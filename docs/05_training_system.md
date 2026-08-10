# Training system: from loss to exact continuation

## Teacher forcing and cross-entropy

During training, all positions are evaluated in parallel even though the model is causal. For
aligned input/target windows:

\[
\mathcal{L}=-\frac{1}{BT}\sum_{b,t}\log p_\theta(y_{b,t}\mid x_{b,\le t}).
\]

This is teacher forcing: the prefix comes from the real sequence, not the model's sampled history.
It makes training parallel over `T`; generation remains sequential.

Perplexity is `exp(mean NLL)` under a fixed tokenizer and evaluation convention. It should not be
compared casually across different tokenizations or target masks.

## One optimizer step

The baseline loop performs:

1. sample `(input_ids, targets)` with `[B,T]`;
2. forward under the configured attention backend and autocast policy;
3. divide loss by gradient-accumulation count;
4. backward each microbatch;
5. unscale FP16 gradients when GradScaler is enabled;
6. clip global gradient norm;
7. AdamW step and scaler update;
8. evaluate/save only at configured boundaries.

Dividing each microbatch loss is essential: without it, accumulation multiplies the effective
gradient by the number of microbatches unless learning rate is adjusted deliberately.

## Gradient accumulation

With microbatch `B` and `K` accumulation steps, the nominal effective batch is `B×K` windows. It
reduces activation memory relative to a single large batch but launches more work and may change
optimizer behavior if dropout, normalization, or loss reduction differs.

This model uses per-token RMSNorm, so no batch-statistic mismatch exists. Dropout masks still differ
across microbatches, as intended.

## Mixed precision

- FP32 is the CPU/default correctness baseline.
- BF16 keeps an eight-bit exponent and usually does not require loss scaling.
- FP16 has a narrower exponent range; CUDA GradScaler is enabled for FP16.
- TF32 is a CUDA matmul policy, not a tensor storage dtype.

Autocast changes kernel choice and accumulation behavior. Numeric equivalence needs dtype-specific
tolerances, and a faster low-precision run is not equivalent evidence if loss diverges.

## Validation and test responsibilities

Validation uses a dedicated batch sampler whose saved initial RNG state is restored before each
evaluation. Every checkpoint comparison therefore sees the same validation windows.

The best checkpoint is selected by validation loss. The test split is evaluated after the training
loop and is not used to select the checkpoint or tune hyperparameters. The current smoke result also
reports the last-model test loss explicitly; future reports should distinguish best-model and
last-model test metrics if both are evaluated.

## Why resume is more than model weights

Exact continuation needs the complete state transition:

- model parameters;
- AdamW `step`, first moment, and second moment for every parameter;
- optimizer param groups and hyperparameters;
- GradScaler state for FP16;
- global CPU/CUDA Torch RNG states for dropout and stochastic kernels;
- batch sampler RNG state;
- current step, best metric/step, history, source hash, tokenizer fingerprint, and config contract.

If any item is missing, training can continue but it is not the same trajectory.

## Pickle-free checkpoint layout

```text
checkpoints/resume/
  model.safetensors
  model.json
  optimizer.safetensors
  optimizer.json
  rng.safetensors
  training.json
```

Tensor files use safetensors. JSON carries bounded scalar/list metadata and SHA-256 digests. Loading
rejects architecture, hash, source, tokenizer, or training-contract mismatches. `max_steps` may be
increased; other training fields remain fixed.

`training.json` also repeats the model and optimizer tensor hashes. Loading cross-checks these
bundle-level values against `model.json` and `optimizer.json`, and rejects non-local tensor
filenames. This catches accidentally mixed checkpoint files before any state is installed.

The manifest is written last. An interruption can leave an invalid partial new bundle, which fails
hash validation rather than silently loading mixed state. A future version may use versioned
checkpoint directories plus an atomic pointer for rollback to the previous complete step.

## Evidence

The regression suite compares:

```text
seed 23, dropout 0.1, AdamW
uninterrupted steps 1..4
versus
steps 1..2 -> save -> reconstruct process state -> restore -> steps 3..4
```

Every final parameter is bitwise equal; history and test loss are equal. This proves the tiny CPU
implementation and serialized state contract. It does not promise bitwise equality across different
PyTorch versions, devices, kernels, or nondeterministic distributed collectives.

The checked-in [RTX 5060 BF16 smoke artifact](../results/training/rtx5060_bf16_smoke.json) records a
real five-step CUDA run with fused AdamW, gradient accumulation, validation, last-test evaluation,
best-checkpoint reload, and CUDA generation. Validation loss changed from `3.9754` at step zero to
`3.5510` at step five. The corpus and run are far too small for a quality conclusion; the generated
continuation was repetitive and is retained in the artifact rather than presented selectively.

## Future system work

- gradient checkpointing: trade recomputation for activation memory;
- cosine/warmup scheduler with serialized state;
- data-shard cursor and worker RNG for streaming loaders;
- DDP/FSDP state and distributed checkpoint semantics;
- compile-safe optimizer/model boundaries;
- versioned checkpoint garbage collection and retention policy.
