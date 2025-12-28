# Targeted-UNDO: Interpretability-Guided Unlearning in LLMs 🧠🛡️

This repository contains the implementation of **Targeted-UNDO**, a research project for the course [Interpretability of Large Language Models](https://moodle.tau.ac.il/) at Tel Aviv University, Fall 2025/2026.

## Overview 📖
Our goal is to enhance the **UNDO** (Unlearn-Noise-Distill-on-Outputs) method by incorporating mechanistic interpretability to localize specific knowledge for more focused noise in the Noise step. 

This project is based on the framework introduced in the paper:
> **[Distillation Robustifies Unlearning](https://arxiv.org/abs/2506.06278)** (2025)

## Motivation 🚀
Standard unlearning methods often require global parameter changes, which can be computationally expensive and may degrade unrelated model capabilities. By identifying specific components (neurons, layers, or heads) responsible for undesired behaviors using **Sparse Autoencoders (SAEs)**, we can apply noise more "surgically." This approach aims to improve both computational efficiency and the robustness of the unlearning.

## Key Features 🛠️
* **Localization Pipeline**: Mapping harmful concepts from the **WMDP dataset** to specific model features using SAEs.
* **Targeted Noise Injection**: A refined UNDO step that applies noise to localized components rather than the entire model.
* **Comparative Evaluation**: Benchmarking against global UNDO and classic unlearning methods on both *forget* (WMDP) and *retain* (Language/Arithmetic) sets.

## Project Structure 📁
* `/src`: Core implementation of Targeted-UNDO and localization scripts.
* `/data`: Scripts for processing the WMDP and retain datasets.
* `/notebooks`: Exploratory analysis of SAE features and activation patching.
* `/results`: Evaluation metrics, plots, and logs.

## Setup & Installation ⚙️
*(We should update it by our progress)*

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Model Weights**: Download **Gemma-2-2B** weights and corresponding SAE features using the `SAELens` library.

## Team 👥
* **Shir Rashkovits**
* **Omer Hagage**
* **Daya Matok Gawi**

---
Instructor: Dr. Mor Geva | TA: Daniela Gottesman