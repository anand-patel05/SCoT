
# SCoT: Similarity-guided Conflict-aware Task Consolidation for Continual VQA

<p align="center">
  <a href="https://eccv.ecva.net/virtual/2026/poster/4496">
    <img src="https://img.shields.io/badge/ECCV-2026-blue" alt="ECCV 2026">
  </a>
  <a href="https://anand-patel05.github.io/SCoT/">
    <img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page">
  </a>
</p>

<p align="center">
  Anand Patel,
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

## 📌 Abstract

Continual learning in visual question answering (VQACL) requires a single vision–language model to acquire new multimodal reasoning skills from a task stream while retaining prior capabilities. However, naïve sequential finetuning suffers from catastrophic forgetting. Existing continual VQA methods primarily rely on replay or parameter regularization but largely overlook how task-specific updates accumulate and interact in parameter space, particularly whether successive updates are synergistic or conflicting across layers.To address this, we introduce SCoT (Similarity-guided Conflict-aware Task Consolidation), a continual learning framework that represents each task as a parameter update relative to a pretrained anchor model and integrates tasks through layer-wise parameter-space reasoning. For each layer, SCoT measures alignment between incoming and accumulated task vectors, removes only destructive components via conditional projection when conflicts arise, and adaptively modulates consolidation strength using similarity-guided weighting. This preserves beneficial transfer while suppressing harmful interference, enabling stable yet adaptive continual learning. Experiments on VQAv2 and NExT-QA demonstrate strong continual VQA performance, reducing forgetting to near-zero (0.07 and -1.90) while achieving rare positive backward transfer (+5.64 and +6.97), outperforming strong continual-learning and task-vector baselines.

For additional details, please refer to our [project page](https://anand-patel05.github.io/SCoT/).

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/anand-patel05/SCoT.git
cd SCoT
```

### 2. Set up the environment

Please refer to the [VQACL repository](https://github.com/zhangxi1997/VQACL) for the original environment configuration and dependencies.

For example:

```bash
conda create -n scot python=3.10
conda activate scot

pip install -r requirements.txt
```

Make sure that the installed PyTorch, CUDA, and other dependencies are compatible with your GPU and the released implementation.

### 3. Download the pretrained backbone

SCoT uses the VL-T5 backbone for continual VQA experiments.

Please follow the pretrained backbone download instructions provided in the official VQACL repository:

**[VQACL — Setup and Backbone Download](https://github.com/zhangxi1997/VQACL#setup)**

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

# Training with 1 gpu for VQA v2
cd VL-T5/
bash scripts/VQACL_train.sh 1 # Standard Training
bash scripts/VQACL_COMP_train.sh 1 # Training for Novel Composition Testing (Group-1)

# Testing with 1 gpu for VQA v2
cd VL-T5/
bash scripts/VQACL.sh 1 # Standard Testing
bash scripts/VQACL_COMP.sh 1 # Novel Composition Testing (Group-1)

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
