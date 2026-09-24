
# SCoT: Similarity-guided Conflict-aware Task Consolidation for Continual VQA

<p align="center">
  <a href="https://eccv.ecva.net/Conferences/2026/AcceptedPapers">
    <img src="https://img.shields.io/badge/ECCV-2026-blue" alt="ECCV 2026">
  </a>
  <a href="https://anand-patel05.github.io/SCoT/">
    <img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page">
  </a>
</p>

<p align="center">
  <a href="https://anand-patel05.github.io/">Anand Patel</a>,
  Moloud Abdar,
  Biplab Banerjee
</p>

<p align="center">
  <strong>European Conference on Computer Vision (ECCV), 2026</strong>
</p>

<p align="center">
  <a href="https://anand-patel05.github.io/SCoT/">Project Page</a>
  |
  <a href="https://eccv.ecva.net/Conferences/2026/AcceptedPapers">ECCV 2026</a>
</p>

This repository provides the official PyTorch implementation of **SCoT: Similarity-guided Conflict-aware Task Consolidation for Continual VQA**, accepted at ECCV 2026.

## 📌 Overview

Continual Visual Question Answering (VQACL) requires a vision-language model to sequentially acquire new reasoning capabilities while preserving previously learned knowledge.

However, sequential fine-tuning often leads to catastrophic forgetting due to interference between task-specific parameter updates.

We propose **SCoT**, a task-vector-based consolidation framework that explicitly models the interaction between incoming and previously accumulated task updates.

SCoT introduces three key components:

- **Layer-wise Similarity Estimation:** Measures the alignment between incoming and accumulated task vectors.
- **Conflict-aware Projection:** Conditionally removes conflicting components of incoming task vectors to mitigate destructive interference.
- **Adaptive Task Consolidation:** Dynamically adjusts consolidation weights according to layer-wise similarity, balancing knowledge retention and adaptation.

SCoT enables effective continual learning while mitigating catastrophic forgetting and promoting positive backward transfer.

For additional details, please refer to our [project page](https://anand-patel05.github.io/SCoT/).

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone <SCOT_REPOSITORY_URL>
cd SCoT
```

### 2. Set up the environment

Please refer to the [VQACL repository](https://github.com/zhangxi1997/VQACL) for the original environment configuration and dependencies.

For example:

```bash
conda create -n scot python=3.7
conda activate scot

pip install -r requirements.txt
```

Make sure that the installed PyTorch, CUDA, and other dependencies are compatible with your GPU and the released implementation.

### 3. Download the pretrained backbone

SCoT uses the VL-T5 backbone for continual VQA experiments.

Please follow the pretrained backbone download instructions provided in the official VQACL repository:

**[VQACL — Setup and Backbone Download](https://github.com/zhangxi1997/VQACL#setup)**

If the corresponding download script is included in this repository, the backbone can be downloaded using:

```bash
python download_backbones.py
```

Ensure that the downloaded checkpoints are placed in the appropriate directories before starting the experiments.

## 📂 Dataset Preparation

We evaluate SCoT on two continual VQA benchmarks:

- **VQAv2:** Continual learning across different question types and visual concepts.
- **NExT-QA:** Continual learning on video question answering.

We follow the continual learning task partitions and experimental settings introduced in VQACL.

### Download datasets and features

Please refer to the official VQACL repository for detailed instructions and download links for:

1. VQAv2 continual learning task partitions.
2. NExT-QA continual learning task partitions.
3. COCO images and visual features.
4. NExT-QA video features.
5. Required pretrained model checkpoints.

**[VQACL — Dataset Preparation and Model Checkpoints](https://github.com/zhangxi1997/VQACL#dataset-preparation--model-checkpoint)**

After downloading the required resources, organize them according to the directory structure expected by the implementation.

> **Note:** Dataset preparation, task partitioning, and backbone initialization follow the VQACL benchmark. Please refer to the original repository for additional setup details.

## 🚀 Training and Evaluation

SCoT is evaluated under the standard continual VQA and novel composition settings.

### VQAv2

We consider two evaluation settings:

- **Standard Testing:** Evaluates the model's ability to retain previously learned knowledge while acquiring new tasks.
- **Novel Composition Testing:** Evaluates generalization to previously unseen combinations of visual concepts and reasoning skills.

The experiments follow the VQACL continual learning task sequence.

Training involves sequential task adaptation followed by similarity-guided, conflict-aware task-vector consolidation.

### NExT-QA

SCoT is also evaluated on NExT-QA to assess its effectiveness in continual video question answering.

The experimental setup follows the corresponding VQACL task partitions.

### Running experiments

Please use the training and evaluation scripts provided in this repository.

<!--
Add the exact commands from the released SCoT implementation here.

For each dataset, document:
- Standard training
- Novel composition training
- Standard evaluation
- Novel composition evaluation
- Replay memory configuration
- Checkpoint and output directories
-->

## 📊 Results

SCoT is evaluated using Final Average Performance (AP), Average Forgetting (AF), and Backward Transfer (BWT).

Our experiments on VQAv2 and NExT-QA demonstrate effective knowledge consolidation with reduced forgetting and positive backward transfer.

For detailed quantitative comparisons, ablation studies, and experimental analysis, please refer to our paper and [project page](https://anand-patel05.github.io/SCoT/).

## 📝 Citation

If you find SCoT useful for your research, please consider citing our paper:

```bibtex
@inproceedings{patel2026scot,
  title     = {SCoT: Similarity-guided Conflict-aware Task Consolidation for Continual VQA},
  author    = {Patel, Anand and Abdar, Moloud and Banerjee, Biplab},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## 🙏 Acknowledgements

This implementation builds upon the following open-source research repositories:

- [VQACL (CVPR 2023)](https://github.com/zhangxi1997/VQACL): Provides the continual VQA benchmark, task partitions, dataset preparation instructions, pretrained backbone setup, and experimental framework.

- [QUAD (ICCV 2025)](https://github.com/IemProg/QUAD): Provides a reference implementation for continual visual question answering and associated experimental components.

We sincerely thank the authors of VQACL and QUAD for making their code and resources publicly available.

Please refer to the original repositories for their respective licenses, dependencies, and citation information.
