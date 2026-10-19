# GPU-Accelerated 2D Heat Diffusion: From Jacobi & Red-Black SOR to Geometric Multigrid

[![CUDA](https://img.shields.io/badge/CUDA-12.x%20%7C%2013.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Architecture](https://img.shields.io/badge/GPU-NVIDIA%20Tesla%20T4%20(Turing%20sm__75)-76B900.svg)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![IEEE Report](https://img.shields.io/badge/Paper-IEEE%20Format-blue.svg)](G25ait2026_IEEE_report_final.pdf)
[![Implementation](https://img.shields.io/badge/Colab-Jupyter%20Notebook-orange.svg)](G25ait2026_final.ipynb)

Comprehensive CUDA C++ implementation, numerical analysis, and architectural optimization of 2D steady-state heat diffusion ($\nabla^2 T = 0$) on an $N \times N$ Dirichlet domain. This repository tracks the evolution from textbook Jacobi iteration to an optimal **$\mathcal{O}(N^2)$ Geometric Multigrid (GMG)** solver accelerated with **CUDA Graphs** and **fused device-side convergence detection**.

---

## 👥 Authors & Course Information

- **Peteti Charantej** (G25AIT2026)
- **Aryaveer Rathi** (G25AIT2021)
- **Sandeep** (G25AIT2036)
- **Course**: GPU Programming Assignment, Problem 1
- **Faculty Advisor**: Dr. Binod Kumar

---

## 🚀 Key Highlights & Architectural Findings

1. **Host-Sync-Free Convergence Detection**:
   - Updates fuse warp-shuffle reductions (`__shfl_down_sync`) and block-level `atomicMax` on IEEE-754 bit patterns (`atomicMaxNonNeg`).
   - Device-side early exit prevents redundant computation. Host polls an asynchronous 4-byte flag only once every $K=200$ iterations.
   - Grid crosses the PCIe bus **exactly twice** (HtoD on startup, DtoH upon completion).
2. **Algorithm Beats Kernel Tuning**:
   - Switching from Jacobi Over-Relaxation (JOR) to Red-Black Successive Over-Relaxation (RB-SOR) cuts iterations by **145×** at $N=512$ ($1,040$ vs $150,654$).
   - Geometric Multigrid (GMG) reduces the problem to **$\mathcal{O}(1)$ iterations** (converges in **3 V-cycles** across all grid sizes from $N=129$ to $N=4097$).
   - At $N=1025$, GMG finishes in **8.35 ms** vs **442.7 ms** for RB-SOR (**53× speedup**) and **~57 s** for JOR (**~6800× speedup**).
3. **Global Memory vs Shared-Memory Tiling on Turing (T4)**:
   - Shared-memory tiling with halo staging is **10% to 16% slower** than global memory.
   - Because Turing merges L1 cache and shared memory into the same 96 KB SRAM array, L1 already captures 5-point stencil reuse automatically. Staging tiles explicitly introduces syncthreads barriers, 33% halo load overhead, and higher register pressure.
4. **Stopping-Rule Accuracy & Floating-Point Precision**:
   - The standard stopping criterion $\Delta^k = \|T^{k+1} - T^k\|_\infty < \epsilon$ measures iterate displacement, not true discrete error. For slowly converging methods with spectral radius $\rho \approx 1 - \mathcal{O}(h^2)$, error scales as $\|e^k\|_\infty \lesssim \frac{\rho}{1-\rho} \Delta^k$. At $N=512$, JOR stops **5.29 °C** away from the true solution, whereas GMG stops within **$4.5 \times 10^{-7}$ °C**.
   - Single precision (`float32`) stalls on a round-off plateau for SOR fine grids because rounding noise is amplified by $\frac{1}{2-\omega^*}$; hence, **FP64 is required** for high accuracy.

---

## 📐 Problem Formulation

The steady-state temperature distribution on a 2D plate $\Omega = [0, 1]^2$ is governed by the Laplace equation:
$$\nabla^2 T = \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0$$

Discretised on an $N \times N$ grid with mesh spacing $h = \frac{1}{N - 1}$:
$$T_{i,j}^{t+1} = \frac{1}{4} \left( T_{i-1,j}^t + T_{i+1,j}^t + T_{i,j-1}^t + T_{i,j+1}^t \right)$$

### Boundary Conditions
- **Top Edge**: $100\ ^\circ\text{C}$
- **Left Edge**: $75\ ^\circ\text{C}$
- **Right Edge**: $50\ ^\circ\text{C}$
- **Bottom Edge**: $0\ ^\circ\text{C}$
- **Interior Initial**: $0\ ^\circ\text{C}$
- Analytical superposition center temperature: $\approx 56.25\ ^\circ\text{C}$ (GPU converged: $56.2\ ^\circ\text{C}$).

---

## 🔬 Solvers Implemented

### 1. Jacobi Over-Relaxation (JOR)
- Stencil update with over-relaxation parameter $\omega$ (for $\omega = 1$, strictly matches standard Jacobi).
- Requires ping-pong buffers (two $N \times N$ grids).
- Spectral radius $\rho_J = \cos(\pi h) \approx 1 - \frac{\pi^2 h^2}{2}$. Diverges for $\omega > \frac{2}{1 + \rho_J} \approx 1.01$.
- Asymptotic convergence: $\mathcal{O}(N^2)$ iterations.

### 2. Red-Black Gauss-Seidel (RB-GS) & Red-Black SOR (RB-SOR)
- Colouring nodes by parity of $(i + j)$ decouples red and black cells.
- Each half-sweep depends only on cells of the opposite colour, allowing in-place single-buffer updates.
- Optimal relaxation parameter (Young's SOR theory):
  $$\omega^* = \frac{2}{1 + \sqrt{1 - \rho_J^2}} = \frac{2}{1 + \sin(\pi h)}, \quad \rho_{\text{SOR}} = \omega^* - 1$$
- Iteration count drops from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$.

### 3. Geometric Multigrid (GMG)
- Multigrid hierarchy: $N_0 = N$, $N_{l+1} = (N_l - 1)/2 + 1$ down to $3 \times 3$ ($N = 2^k + 1$).
- **Smoother**: Red-Black Gauss-Seidel ($\nu_1 = 2$ pre-smoothing, $\nu_2 = 2$ post-smoothing sweeps).
- **Residual**: $r = f - A_h u$.
- **Restriction ($R$)**: Full weighting stencil $\frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$.
- **Prolongation ($P$)**: Bilinear interpolation.
- **Cycles**: $V(2,2)$, $W(2,2)$, and Full Multigrid (FMG) initialization.
- **CUDA Graphs**: Captures the entire multi-level execution pipeline (~100 kernel launches per V-cycle) into a single instantiated graph executable, eliminating CPU launch latency.

---

## 📊 Experimental Results

All experiments evaluated on **Google Colab NVIDIA Tesla T4 GPU** (40 SMs, 16 GB GDDR6, 320 GB/s peak, CUDA 13.0, `nvcc -O3 -arch=sm_75`, FP64 default).

### 1. Iterations & Discretisation Error ($\epsilon = 10^{-4}$, FP64)

| $N$ | JOR Iters | RB-GS Iters | RB-SOR Iters | JOR Error (°C) | RB-SOR Error (°C) | GMG Cycles | GMG Error (°C) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **64** | 5,655 | 3,107 | 144 | $8.0 \times 10^{-2}$ | $8.6 \times 10^{-4}$ | 3 | $< 10^{-6}$ |
| **128** | 18,404 | 10,336 | 276 | 0.33 | $1.6 \times 10^{-3}$ | 3 | $< 10^{-6}$ |
| **256** | 55,830 | 32,483 | 531 | 1.32 | $2.7 \times 10^{-3}$ | 3 | $< 10^{-6}$ |
| **512** | 150,654 | 93,660 | 1,040 | 5.29 | $3.4 \times 10^{-3}$ | 3 | $< 10^{-6}$ |
| **1024** | 311,637 | 228,375 | 2,035 | 21.01 | $4.3 \times 10^{-3}$ | 3 | $4.5 \times 10^{-7}$ |
| **2048** | — | — | 4,014 | — | $5.2 \times 10^{-3}$ | 3 | $< 10^{-6}$ |
| **4096** | — | — | 7,869 | — | $6.8 \times 10^{-3}$ | 3 | $< 10^{-6}$ |

> **Fitted scaling**: JOR $\sim N^{1.46}$, RB-GS $\sim N^{1.56}$, RB-SOR $\sim N^{0.96}$, GMG $\sim \mathcal{O}(1)$ (constant 3 cycles).

---

### 2. GPU Time to Solution ($\epsilon = 10^{-4}$, FP64)

| $N$ | JOR Global (ms) | JOR Shared (ms) | RB-SOR Global (ms) | GMG V(2,2)+FMG (ms) | Speedup (GMG vs RB-SOR) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **129** | 90.7 ($N=128$) | 105.6 | 3.2 | **1.54** | **2.0×** |
| **257** | 666.6 ($N=256$) | 730.6 | 8.4 | **1.94** | **4.3×** |
| **513** | 6,303.8 ($N=512$) | 7,032.9 | 52.5 | **2.99** | **17.6×** |
| **1025** | 56,922 ($N=1024$) | 63,586 | 442.7 | **8.35** | **53.0×** |
| **2049** | — | — | 3,351.8 | **30.14** | **111.2×** |
| **4097** | — | — | 25,766.5 | **120.95** | **213.0×** |

---

### 3. Asymptotic Cost Comparison Summary

| Metric | JOR ($\omega=1$) | RB-SOR ($\omega^*$) | Geometric Multigrid (GMG) |
|:---|:---:|:---:|:---:|
| **Iterations to $\epsilon$** | $\mathcal{O}(N^2)$ | $\mathcal{O}(N)$ | $\mathcal{O}(1)$ |
| **Work per Iteration** | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ ($\approx 9$ Work Units) |
| **Total Algorithmic Work** | $\mathcal{O}(n^2)$ | $\mathcal{O}(n^{1.5})$ | $\mathcal{O}(n)$ (*Optimal*) |
| **Memory Footprint** | $2N^2$ | $N^2$ | $\approx 4N^2$ |
| **Parallel Depth** | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ | $\mathcal{O}(\log N)$ |
| **Measured Runtime ($N=1025$)** | $\sim 57\text{ s}$ | $442.7\text{ ms}$ | **$8.35\text{ ms}$** |
| **Error at $\epsilon=10^{-4}$** | $21.01\ ^\circ\text{C}$ | $4.4 \times 10^{-3}\ ^\circ\text{C}$ | **$4.5 \times 10^{-7}\ ^\circ\text{C}$** |
| **Tuned Parameter** | None | $\omega^* = \frac{2}{1 + \sin(\pi h)}$ | None |

---

## 🛠 Repository Structure

```text
├── Dockerfile                           # Production multi-GPU CUDA Docker container
├── Makefile                             # Build configuration for CUDA solvers & Docker
├── .dockerignore                        # Docker build context exclusions
├── heat2d.cu                            # Standalone CUDA source for JOR and Red-Black SOR
├── gmg.cu                              # Standalone CUDA source for Geometric Multigrid
├── G25ait2026_final.ipynb               # Full Jupyter Notebook with CUDA C++ code & plots
├── G25ait2026_IEEE_report_final.pdf     # Complete IEEE formatted research report
├── G25ait2026_HeatDiffusion_final.pptx  # Presentation deck
├── README.md                            # Comprehensive documentation & benchmarking
```

---

## 💻 How to Run

### Option 1: Docker Container (Recommended for Reproducibility)

Build and run the container with full GPU access across architectures (T4, A100, RTX 30xx/40xx, H100):

```bash
# 1. Build the Docker image
docker build -t gpu-heat-diffusion:latest .

# 2. Run default GMG benchmark inside GPU container
docker run --gpus all --rm -it gpu-heat-diffusion:latest

# 3. Run interactive JupyterLab inside GPU container
docker run --gpus all --rm -it -p 8888:8888 gpu-heat-diffusion:latest \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# 4. Or use make targets
make docker-build
make docker-run
make docker-jupyter
```

### Option 2: Local Compilation with `make`

```bash
# Compile both heat2d and gmg solvers
make

# Run JOR & RB-SOR solver benchmark
./heat2d -n 256 -e 1e-4 -m all -v both -p double

# Run Geometric Multigrid benchmark
./gmg -n 513 -e 1e-4 -c V -p double
```

### Option 3: In Google Colab
1. Upload and open [`G25ait2026_final.ipynb`](G25ait2026_final.ipynb).
2. Set Runtime Type to **GPU (Tesla T4)**.
3. Run the cells sequentially to compile kernels via `nvcc`, execute benchmarks, and generate validation plots.

### Verification Mode
The implementation includes a `--verify` flag executing CPU reference validation:
- Validates bit-for-bit equivalence between sequential CPU execution and GPU kernels using an emulated barrier and warp-shuffle model.
- Checks convergence accuracy across odd/even dimensions, float32, and float64 precisions.
  ```bash
  ./heat2d -n 37 -e 1e-4 --verify
  ./gmg -n 65 -e 1e-4 --verify
  ```

---

## 📖 Citation & References
- D. M. Young, *Iterative Solution of Large Linear Systems*. Academic Press, 1971.
- W. L. Briggs, V. E. Henson, and S. F. McCormick, *A Multigrid Tutorial*, 2nd ed. SIAM, 2000.
- NVIDIA Corporation, *CUDA C++ Programming Guide*, 2024.
- M. Harris, "Optimizing parallel reduction in CUDA," *NVIDIA Developer Technology*, 2007.
