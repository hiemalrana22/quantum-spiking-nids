# QSNN-NIDS — Quantum Spiking Neural Network for Network Intrusion Detection

**B.Tech BCSE497J Project–I · Network Intrusion Detection using QML**
Jayati Jain (23BDS0056) · Piyush Raj (23BDS0105) · Hiemal Rana (23BDS0180)
Guide: Parthiban Krishnamoorthy

---

## 1. What this is, and why it's novel

Every one of the 25 papers in the Review-1 literature table uses a **static-vector**
quantum classifier — QSVM/QSVC, QCNN, HQNN, QGAN, QLSTM, quantum k-means. Quantum
Spiking Neural Networks built on **Quantum Leaky Integrate-and-Fire (QLIF)** neurons
exist, but only for image classification, traffic-sign recognition, fraud and
speech tasks. **No published QSNN for network intrusion detection.**

That is this project's contribution, and it is a defensible one rather than a
kernel swap:

| Gap in the literature table | How a QSNN addresses it |
|---|---|
| Network traffic is a temporal event stream, but QSVM/QCNN treat each flow as a static vector | Spiking networks integrate evidence over discrete time by construction |
| "Not tested against real quantum hardware noise" (papers 1, 9, 14) | QLIF measures every timestep, so **circuit depth is O(ansatz) and constant in T** — it does not deepen with sequence length the way a QLSTM does |
| "Current quantum hardware too small / limited in qubits" (papers 4, 20) | One qubit per neuron; an 8-qubit model is a complete network, not a truncation |
| "Deeper circuits suffer higher noise" (paper 25) | Measure-and-prepare resets after each step, so noise does not accumulate across the sequence |
| Energy/efficiency never quantified | Spiking is sparse — we report spike rate, circuit depth, 2-qubit gate count and latency alongside accuracy |

### The QLIF neuron

One qubit **is** one neuron. Per timestep `t`, for a population of `N` qubits:

1. **Integrate** (classical, cheap)
   `U_t = β · U_{t−1} + I_t − S_{t−1}·v_th`  — leak `β`, refractory reset.
2. **Prepare** (quantum) — membrane → Bloch polar angle
   `θ_t = π · σ(g·U_t + b)` → `RY(θ_t)|0⟩`
   The sigmoid map is **strictly monotone**, so `P(|1⟩)` rises with the membrane
   over the whole real line. (A symmetric map such as `π·tanh(U)` makes a strongly
   *inhibited* neuron fire exactly as hard as an excited one, cancelling the sign of
   every synaptic weight — `angle_map: tanh` reproduces that failure as an ablation.)
3. **Couple** (quantum) — a trainable entangling ansatz mixes the population.
   This is the part with no classical analogue: neurons share *amplitude*, not just
   a weight matrix.
4. **Measure & fire** — `P(|1⟩_i) = (1 − ⟨Z_i⟩)/2`, spike if it exceeds `v_th`.
   Measurement collapses the qubit; the next step re-prepares from the classical
   membrane. This **measure-and-prepare** scheme is what keeps depth constant in `T`.

The hard spike is non-differentiable, so training uses a **surrogate gradient**
(fast-sigmoid / atan / sigmoid) — exact 0/1 forward, smooth backward.

```
flow features x ∈ [0,1]^F
  → SpikeEncoder            [B,T,F]   rate | latency | phase | delta | repeat
  → Linear synapse F→Q      [B,T,Q]   classical, shared across timesteps
  → QuantumLIFLayer (Q qubits)        spikes [B,T,Q]
  → Linear synapse Q→Q  →  QuantumLIFLayer   (deep variant)
  → readout (spike rate ‖ final membrane ‖ mean P(|1⟩))
  → Linear → logits         [B,C]
```

---

## 2. Datasets

Both are used, as the research note recommends: one classic for direct
comparability with the literature table, one broader/modern set to answer the
"outdated dataset" objection.

### UNSW-NB15 — primary
<https://research.unsw.edu.au/projects/unsw-nb15-dataset>

42 flow features, 9 attack families, ~257k rows in the pre-split files. Used by 8
of the 25 surveyed papers, so the numbers are directly comparable
(QSVC 99.78 %, VQNN-QCNN 94.51 %, QMGOA 99.89 %).

```
data/raw/unsw_nb15/
├── UNSW_NB15_training-set.csv
└── UNSW_NB15_testing-set.csv
```

### CIC-IDS2017 — second / cross-dataset benchmark
<https://www.unb.ca/cic/datasets/ids-2017.html>

5 capture days, ~2.8M CICFlowMeter flows, 78 numeric features, 14 attack labels.
Timestamped, so it also drives the **sequential window mode** where `T` is real
traffic time rather than an encoding axis. Comparability point: the hybrid QLSTM
paper (97.43 %). Download the `MachineLearningCVE/` CSV folder.

```
data/raw/cicids2017/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
└── ... (8 files)
```

Training on one and testing on the other is the **cross-dataset generalisation**
experiment that papers 10 and 17 flag as an open gap.

### Preprocessing (identical for every model, so comparisons are fair)

Drop identity/leakage columns (IPs, ports, timestamps — otherwise the model
memorises the capture schedule) → clean `inf`/`NaN`/duplicates → one-hot the
categoricals (`proto`, `service`, `state`) → **split first**, then fit every
transform on train only → quantile scaling (robust to byte/duration heavy tails)
→ mutual-information filter to 32–40 features → PCA (or autoencoder) to
`n_qubits` dims → min-max to `[0,1]` for angle/spike encoding.

### Choosing the spike code

Encoding is the one design choice that measurably dominates everything else, so
it is worth getting right before burning cloud budget. Measured on the offline
fixture — fit a Random Forest on the raw features, then on the *spike-rate decode*
of each encoder, and compare AUC:

| Encoder | T=8 | T=32 | T=128 | |
|---|---|---|---|---|
| raw features (ceiling) | 0.879 | — | — | |
| **phase** | **0.845** | **0.880** | 0.883 | deterministic, near-lossless at small T |
| **delta** | 0.860 | 0.881 | 0.882 | deterministic, sparsest of the three |
| rate | 0.722 | 0.812 | 0.868 | Bernoulli variance; needs T≈128 to catch up |
| latency | 0.530 | 0.500 | 0.500 | — see below |

So `phase` is the default. Bernoulli `rate` coding throws away ~18 % of the
separability at T=8 — each feature is estimated from only 8 coin flips — and
recovering it costs 16× more circuit evaluations.

`latency` (time-to-first-spike) scores at chance **in this table only because the
probe decodes spike *counts***: TTFS puts exactly one spike per feature, so the
count is constant and the information lives entirely in *when* it fires. The QSNN
itself reads it correctly through the membrane's temporal integration. Judge
`latency` from `scripts/ablation.py --study encoding`, not from this table.

---

## 3. Install & verify

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the offline self-check first — **synthetic data, no download needed**. It
verifies preprocessing, all five spike encoders, the QLIF forward pass, that
gradients actually reach the variational circuit parameters, membrane dynamics,
the training loop, the noisy device, and the baselines:

```bash
python scripts/smoke_test.py
```

---

## 4. Running it

```bash
# 1. cache a preprocessed bundle (do this once, reuse everywhere)
python scripts/prepare_data.py --config configs/unsw_nb15.yaml

# 2. train + evaluate + all baselines on the same split
python scripts/train.py --config configs/unsw_nb15.yaml \
    --bundle data/processed/unsw_nb15.npz

# 3. the experiment ~60% of the surveyed papers skip
python scripts/eval_noise.py --config configs/unsw_nb15.yaml \
    --checkpoint artifacts/qsnn_unsw/best.pt \
    --levels 0 0.005 0.01 0.02 0.05 0.1

# 4. ablations reviewers will ask for
python scripts/ablation.py --config configs/unsw_nb15.yaml --study encoding
python scripts/ablation.py --config configs/unsw_nb15.yaml --study qubits
python scripts/ablation.py --config configs/unsw_nb15.yaml --study quantum_vs_classical
```

Useful overrides: `--qubits 10 --timesteps 16 --encoding phase --noise
--classical-lif --epochs 40 --baselines rf svm mlp qsvc qcnn`.

---

## 5. Baselines — run on the identical pipeline

Most surveyed papers quote accuracy figures from *other* papers, produced on
different splits and different feature pipelines. Here every baseline consumes
the exact same `DataBundle`:

| | Model | Notes |
|---|---|---|
| Classical | Random Forest, RBF-SVM, MLP | PPT objective 4 |
| Quantum | **QSVC** (ZZ feature-map kernel + SVM) | the dominant model in the review table |
| Quantum | **QCNN** (Cong-style conv/pool, SU(4) unitaries) | papers 5, 8, 18 |
| Ablation | classical-LIF SNN (`--classical-lif`) | isolates what the *qubit* buys |

---

## 6. Metrics

Detection: accuracy, balanced accuracy, precision, recall, F1, **FPR** (the metric
that actually decides analyst alert-fatigue and deployability), MCC, ROC-AUC,
PR-AUC, full confusion matrix.

Efficiency — the differentiator, since **none** of the 25 surveyed papers reports
these together: qubit count, circuit depth per timestep (measured on the
*decomposed* circuit via `qml.specs(level="device")`), two-qubit gate count,
circuit evaluations per sample, quantum vs classical parameter split, inference
latency (ms/flow), and **spike rate** (the sparsity/energy proxy).

---

## 7. Noise & real hardware

`configs/*.yaml → noise:` selects the device.

- `analytic` — density-matrix simulation (`default.mixed`) with depolarizing +
  amplitude/phase damping. Exact channels, still backprop-differentiable, so you
  can **train** under noise, not just evaluate.
- `qiskit_aer` — Qiskit Aer noise model with shot sampling and readout error.
  Automatically falls back to parameter-shift gradients.
- `ibm_runtime` — real IBM hardware (`pennylane-qiskit` + credentials).

```yaml
noise: {enabled: true, backend: analytic, depolarizing: 0.01, amplitude_damping: 0.005}
```

---

## 8. Where to train it (measured, not guessed)

**The decisive fact: this workload is single-core CPU-bound.** An 8-qubit state
vector is 256 complex amplitudes — far too small to use a GPU or extra cores.
Measured on an 8-core M-series Mac at the default config (8 qubits, T=12, 2 QLIF
layers = 24 circuit evaluations per sample):

| threads | 1 | 4 | 8 |
|---|---|---|---|
| ms/sample | 9.6 | 10.2 | 10.0 |

Flat. **Do not go hunting for a GPU or a many-core box — neither speeds up a
single run.** More cores only help you run *independent* configs (ablations)
side by side.

Cost at batch 256 (`7.4 ms/sample/epoch`):

| train rows | per epoch | 30 epochs |
|---|---|---|
| 20,000 | 2.5 min | **1.2 h** |
| 50,000 | 6.1 min | **3.1 h** |
| 100,000 | 12.3 min | **6.1 h** |

Cost scales linearly with `timesteps × n_qlif_layers`. Halving to `T=8`,
`n_qlif_layers: 1` cuts the bill 3×.

### Free platforms, best first

1. **Kaggle Notebooks** — the right choice. 12 h sessions, 30 h/week, and *"Save &
   Run All" executes headless in the background* so you don't babysit a browser
   tab. Both UNSW-NB15 and CIC-IDS2017 are already hosted as public Kaggle
   datasets, so you skip the download and registration entirely.
2. **Your own laptop** — genuinely competitive here, because the job is
   single-threaded. An 8-core machine runs 6–8 ablation configs *simultaneously*
   at full speed, which Kaggle's single session cannot.
3. **Google Colab (free)** — works, but weaker: 2 vCPUs, and it disconnects after
   ~90 min idle unless the tab stays active. Fine for `smoke_test.py` and short
   runs; poor for a 3 h job.
4. **GitHub Codespaces** — 60–120 free core-hours/month on a 2-core box, runs
   headless. A reasonable third option.
5. **IBM Quantum (Open Plan)** — free monthly QPU time. *Not* for training, but
   enough for the real-hardware validation run (`noise.backend: ibm_runtime`) that
   would put this ahead of ~60 % of the papers in the review table.

### Budget plan that fits one free session

```bash
python scripts/smoke_test.py                       # verify env first (~1 min)
python scripts/prepare_data.py --config configs/unsw_nb15.yaml
python scripts/train.py --config configs/unsw_nb15.yaml \
    --bundle data/processed/unsw_nb15.npz --epochs 30
```

Set `max_rows: 50000` for a ~3 h QSNN run, leaving headroom in a 12 h session for
the baselines. Watch the QSVC baseline: its quantum kernel is **O(n²) circuits** at
~9 ms each, so `qsvc_max_train` costs ~0.6 h at 500, ~2.5 h at 1000 and ~10 h at
2000. It ships capped at 500 for exactly this reason.

## 9. Layout

```
qsnn_nids/
├── configs/           unsw_nb15.yaml · cicids2017.yaml · smoke.yaml
├── qsnn/
│   ├── config.py          dataclass config, YAML + CLI overrides
│   ├── noise.py           device factory: ideal / mixed / Aer / IBM
│   ├── metrics.py         detection + quantum-resource metrics
│   ├── engine.py          train loop, focal loss, early stopping
│   ├── data/
│   │   ├── datasets.py        UNSW-NB15 + CIC-IDS2017 registry
│   │   ├── preprocess.py      cleaning → scaling → MI → PCA/AE → [0,1]
│   │   └── spike_encoding.py  rate · latency · phase · delta · repeat · windows
│   └── models/
│       ├── qlif.py            QuantumLIFLayer (+ classical-LIF control)
│       ├── qsnn.py            the full model
│       ├── surrogate.py       surrogate-gradient spike functions
│       └── baselines.py       RF · SVM · MLP · QSVC · QCNN
└── scripts/           prepare_data · train · eval_noise · ablation · smoke_test
```

## 10. Status, and two things to know before you train

The model, pipeline, baselines, noise simulation and ablations are implemented and
verified end-to-end by `scripts/smoke_test.py` (11/11 checks pass on synthetic
data), and all five entry points run clean. **No results have been produced on
UNSW-NB15 or CIC-IDS2017** — training is intended to run on cloud resources. Every
accuracy number in the write-up must come from an actual run.

Two design faults were found and fixed while bringing the model up, both worth
knowing because the naive version of each is what most tutorial QSNN code does:

1. **Symmetric angle map.** Mapping the membrane with `θ = π·tanh(U)` makes
   `P(|1⟩)` symmetric about `U=0`, so a strongly inhibited neuron fires exactly as
   hard as an excited one and the sign of every synaptic weight is cancelled. The
   network sat at chance. Fixed with the monotone `π·σ(g·U+b)` map; `angle_map:
   tanh` keeps the broken version available as an ablation.
2. **LayerNorm on the readout.** It subtracts the per-sample mean, which *is* the
   population firing rate — i.e. it deleted the signal. Replaced with BatchNorm.

On the offline fixture these two fixes plus `phase` coding moved test ROC-AUC from
0.50 (chance) to 0.70–0.76. That confirms the architecture learns; it is **not**
evidence that it beats the classical baselines. On that same tiny fixture Random
Forest still reaches 0.88, and the QSNN is visibly underfit at 6 qubits / 6 epochs
on ~1.7k synthetic rows. Expect to tune `n_qubits`, `timesteps`, `ansatz_layers`
and the learning rates on the real datasets before drawing any comparison.
