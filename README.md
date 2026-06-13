# LonghornAI

A portable, pure **C++17** library of the common AI-inference kernels found
across modern LLM architectures (Llama, Mistral, Qwen, Gemma, Phi, ...). No
CUDA, no GPU code, no third-party compute dependencies — the same kernels build
and run identically on Windows, Linux, and macOS with MSVC, GCC, or Clang.

See [`PLAN.md`](PLAN.md) for the staged roadmap and design rationale.

Every kernel ships with a naive reference (defining the numerics), a faster
implementation, and a test that proves they agree within a documented
tolerance. The sections below list each kernel with the math it computes.

## Build & test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Optional multithreading (never required to build): `-DLONGHORN_ENABLE_OPENMP=ON`.

## Layout

```
src/kernels/   one <name>.hpp + <name>.cpp per kernel family
tests/         GoogleTest suite, kernel-vs-reference comparisons
```

---

## Kernels

### GEMM — `src/kernels/gemm.hpp`

General matrix multiply with FP32 accumulation, plus batched and grouped
variants.

$$C = \alpha \, A B + \beta \, C, \qquad C_{ij} = \alpha \sum_{k=1}^{K} A_{ik} B_{kj} + \beta \, C_{ij}$$

Batched (uniform shapes): $C_b = \alpha A_b B_b + \beta C_b$ for $b = 1 \dots N$.

Grouped (ragged shapes): $C_g = \alpha A_g B_g + \beta C_g$, where each group $g$
has its own $(M_g, N_g, K_g)$.

### LayerNorm — `src/kernels/normalization.hpp`

Normalize over the last dimension with learnable scale $\gamma$ and shift
$\beta$.

$$\mu = \frac{1}{D}\sum_{i=1}^{D} x_i, \qquad \sigma^2 = \frac{1}{D}\sum_{i=1}^{D} (x_i - \mu)^2$$

$$y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \, \gamma_i + \beta_i$$

### RMSNorm — `src/kernels/normalization.hpp`

Root-mean-square normalization (no mean subtraction); used by Llama/Qwen/Mistral.

$$y_i = \frac{x_i}{\sqrt{\frac{1}{D}\sum_{j=1}^{D} x_j^2 + \epsilon}} \, \gamma_i$$

### Softmax — `src/kernels/softmax.hpp`

Numerically stable row softmax (max-subtracted).

$$y_i = \frac{e^{\,x_i - m}}{\sum_{j=1}^{D} e^{\,x_j - m}}, \qquad m = \max_j x_j$$

### GELU — `src/kernels/activation.hpp`

Exact (erf) and tanh-approximation variants.

$$\mathrm{GELU}(x) = x \, \Phi(x) = \tfrac{1}{2} x \left(1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right)$$

$$\mathrm{GELU}_{\tanh}(x) = \tfrac{1}{2} x \left(1 + \tanh\!\left[\sqrt{\tfrac{2}{\pi}} \left(x + 0.044715\, x^3\right)\right]\right)$$

### SiLU / Sigmoid — `src/kernels/activation.hpp`

$$\sigma(x) = \frac{1}{1 + e^{-x}}, \qquad \mathrm{SiLU}(x) = x \, \sigma(x)$$

### SwiGLU / GeGLU — `src/kernels/activation.hpp`

Gated MLP activations. The input is split into a gate $a$ and a value $b$.

$$\mathrm{SwiGLU}(a, b) = \mathrm{SiLU}(a) \odot b, \qquad \mathrm{GeGLU}(a, b) = \mathrm{GELU}(a) \odot b$$

### Reductions — `src/kernels/reduction.hpp`

Last-dimension reductions.

$$\mathrm{sum}_r = \sum_{i=1}^{D} x_{r,i}, \qquad \mathrm{max}_r = \max_i x_{r,i}, \qquad \mathrm{mean}_r = \frac{1}{D}\sum_{i=1}^{D} x_{r,i}$$

### RoPE — `src/kernels/rope.hpp`

Rotary position embeddings. For rotated pair index $i \in [0, d/2)$ at position
$p$, with per-pair frequency $\theta_i = \text{base}^{-2i/d}$ and angle
$\phi = p \, \theta_i$:

$$\begin{pmatrix} x'_a \\ x'_b \end{pmatrix} = \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix} \begin{pmatrix} x_a \\ x_b \end{pmatrix}$$

Interleaved layout pairs $(x_{2i}, x_{2i+1})$; half-rotation layout pairs
$(x_i, x_{i + d/2})$. Linear/NTK scaling adjusts $p$ or the base.

### Embedding lookup — `src/kernels/embedding.hpp`

Gather rows from a table by token id, with optional scale $s$.

$$\mathrm{out}_{t,:} = s \cdot \mathrm{table}_{\,\mathrm{ids}[t],\,:}$$

### Scaled Dot-Product Attention (SDPA) — `src/kernels/attention.hpp`

The reference attention; correctness anchor for the flash variant.

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d}} + M\right) V$$

where $M$ is the (optional) causal mask, $M_{ij} = -\infty$ for $j > i$ (aligned
positions), else $0$.

### FlashAttention-style — `src/kernels/attention.hpp`

Same result as SDPA, computed with a tiled **online softmax** that never
materializes the score matrix. Per key block, with running max $m$ and running
normalizer $\ell$:

$$m^{\text{new}} = \max(m,\ \max_j s_j), \qquad \ell \leftarrow \ell \, e^{\,m - m^{\text{new}}} + \sum_j e^{\,s_j - m^{\text{new}}}$$

$$O \leftarrow O \, e^{\,m - m^{\text{new}}} + \sum_j e^{\,s_j - m^{\text{new}}}\, V_j, \qquad O \leftarrow O / \ell$$

### MHA / MQA / GQA — `src/kernels/attention.hpp`

Head configurations selected by the number of key/value heads $H_{kv}$ relative
to query heads $H_q$:

$$\text{MHA: } H_{kv} = H_q, \qquad \text{MQA: } H_{kv} = 1, \qquad \text{GQA: } H_q \bmod H_{kv} = 0$$

Query head $h$ reads key/value head $\lfloor h / (H_q / H_{kv}) \rfloor$.

### KV cache — `src/kernels/attention.hpp`

Append new keys/values for $s_{\text{new}}$ timesteps into a per-head cache at
offset $p$ (the past length), so decode attends over the growing sequence:

$$\mathrm{cache}_{h,\, p + t,\, :} = k^{\text{new}}_{h,\, t,\, :}, \qquad t = 0 \dots s_{\text{new}} - 1$$

---

## Low-precision types — `src/kernels/dtypes.hpp`

Portable software `half` (IEEE-754 binary16) and `bfloat16` with
`to_float`/`from_float` conversions. FP32 is the source of truth; these exist
for storage and for exercising the reduced-precision numeric paths.

## Out of scope

CUDA/GPU backends, training/backward, quantization, MoE, paged attention,
multi-GPU/distributed, and RTL/FPGA/silicon backends. See [`PLAN.md`](PLAN.md).
