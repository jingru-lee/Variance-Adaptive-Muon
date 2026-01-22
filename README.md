# Variance-Adaptive Muon

[![arXiv](https://img.shields.io/badge/arXiv-2601.14603-b31b1b.svg)](https://arxiv.org/abs/2601.14603v1)

This repository contains the source code and experimental results for the paper **"Variance-Adaptive Muon: Accelerating LLM Pretraining with NSR-Modulated and Variance-Scaled Momentum"**, currently submitted to the **International Joint Conference on Artificial Intelligence (IJCAI 2026)**.

📄 **Paper Link:** [https://arxiv.org/abs/2601.14603v1](https://arxiv.org/abs/2601.14603v1)

## 📝 Abstract

This work introduces **Variance-Adaptive Muon**, a novel optimization framework designed to accelerate Large Language Model (LLM) pretraining. We provide a comprehensive comparative analysis of the proposed **Variance-Adaptive Muon** variants (**Muon-NSR** and **Muon-VS**) against standard baselines (**Muon** and **AdamW**).

## 📂 Code Organization

The repository is organized into the following directories:

* **`NanoGPT/`**: Implementation for NanoGPT experiments.
* **`Suite_A/`**: Codebase for Benchmark Suite A.
* **`Suite_B/`**: Codebase for Benchmark Suite B.
* **`Figures/`**: Contains the experimental result figures and plots presented in the paper.

> **Note:** We have encapsulated the entry commands for both the proposed optimizers and baselines in a `run.sh` script for each task.

## 🚀 Usage & Reproduction

To reproduce the results reported in the paper, navigate to the corresponding task directory and execute the shell script.

For example, to run the Suite_A experiments:

```bash
cd Suite_A
bash run.sh
```

## 📖 Citation

If you find this work useful in your research, please consider citing our paper:

```bash
@misc{li2026varianceadaptivemuonacceleratingllm,
      title={Variance-Adaptive Muon: Accelerating LLM Pretraining with NSR-Modulated and Variance-Scaled Momentum}, 
      author={Jingru Li and Yibo Fan and Huan Li},
      year={2026},
      eprint={2601.14603},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.14603}, 
}
```


