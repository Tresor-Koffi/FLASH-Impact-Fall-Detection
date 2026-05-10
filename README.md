# FLASH: Efficient Impact Fall Detection with Unified Hypergraph State-Space Model

<div align="center">

[![Conference](https://img.shields.io/badge/IEEE%20ICIP-2026-blue)](https://icip2026.exordo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/Code-Coming%20Soon-orange)

</div>

---

> **FLASH** is a novel framework for **impact fall detection** that pinpoints the precise ground-impact moment from skeleton sequences. It integrates **single-matrix biomechanically-grounded hypergraphs** for spatial modeling with **Mamba's selective state-space models** for efficient temporal modeling.

📄 **Paper accepted at IEEE International Conference on Image Processing (ICIP 2026)**  
📅 September 13–17, 2026 — Tampere, Finland

---

## 🔍 Overview

Falls represent a critical public health challenge. FLASH addresses the precise detection of the **impact moment** — the instant the body hits the ground — which is crucial for timely medical intervention.

### Key contributions:
- **Single-matrix hypergraph** representation encoding multi-joint coordination while avoiding dual-incidence redundancy
- **Mamba SSM integration** achieving linear-time temporal modeling vs. transformers' quadratic complexity
- **71.3% FLOPs reduction** and **93.9% faster inference** compared to dual-representation methods
- **Interpretable feedback** through learned joint attention patterns aligned with biomechanical principles

---

## 📊 Results

### Performance on UP-Fall Dataset

| Method | Hypergraph | Acc (%) | F1 (%) | Impact Detection |
|--------|-----------|---------|--------|-----------------|
| ST-GCN | No | 92.16 | 91.31 | No |
| 2s-AGCN | No | 95.10 | -- | No |
| Hyper-GNN | Yes (adaptive) | 89.50 | 95.19 | No |
| Transformer | No | 93.15 | 93.80 | No |
| DistillH-Mamba | Yes (dual + KD) | 97.38 | 97.51 | Yes |
| **FLASH (Ours)** | **Yes (single)** | **95.13** | **95.52** | **Yes** |

### Zero-Shot Cross-Dataset Generalization

| Train | Test | Acc (%) | F1 (%) | AUC (%) |
|-------|------|---------|--------|---------|
| UP-Fall | UMAFall (zero-shot) | 95.83 | 96.15 | 98.98 |
| UMAFall | UMAFall (in-domain) | 97.92 | 97.70 | 99.15 |

### Computational Efficiency

| Model | Memory (MB) | Params (M) | FLOPs (×10⁷) | Time (ms) |
|-------|------------|-----------|--------------|-----------|
| DistillH-Mamba (Dual+KD) | 280.46 | 70.12 | 2.266 | 182.3 |
| Transformer | 341.6 | 85.4 | 4.523 | 245.6 |
| **FLASH (Ours)** | **148.4** | **37.1** | **0.65** | **11.2** |

---

## 🏗️ Architecture

FLASH consists of four stages:

1. **Input** — 3D skeleton sequence with 33 joints per frame
2. **HGCN** — 2-layer hypergraph convolution with 6 biomechanical hyperedges
3. **Mamba Block** — Selective SSM with O(T) complexity for temporal modeling
4. **MTCN** — Multi-scale temporal convolutions (kernels 9, 15, 20) + FC classifier

---

## 📦 Code

> 🚧 **Code will be released soon.**  
> We are currently cleaning and documenting the codebase. Please **star ⭐ the repo** to be notified when the code is available.

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{koffi2026flash,
  title     = {FLASH: Efficient Impact Fall Detection with Unified Hypergraph State-Space Model},
  author    = {Koffi, Tresor Y. and Mourchid, Youssef and Dupuis, Yohan},
  booktitle = {IEEE International Conference on Image Processing (ICIP)},
  year      = {2026},
  address   = {Tampere, Finland}
}
```

---

## 📬 Contact

**Tresor Y. Koffi** — ytkoffi@cesi.fr  
**Youssef Mourchid** — ymourchid@cesi.fr  
**Yohan Dupuis** — ydupuis@cesi.fr  

CESI LINEACT, France
