# Efficient Deep Learning Models for Weather Prediction

This repository contains the codebase and experimental results for my M.Tech Thesis at IIT Guwahati. The research focuses on optimizing and compressing large-scale deep learning models used for meteorological forecasting, aiming to drastically reduce computational overhead and memory footprint without sacrificing predictive accuracy.

## Project Overview

High-resolution weather prediction relies heavily on computationally expensive spatial-temporal models. This project tackles the deployment bottleneck by exploring both generic and application-specific model compression techniques tailored for transformer-based architectures.

## Compression Methodologies

### 1. Precision Quantization
To reduce the raw memory footprint of the models, we implemented numerical precision reduction techniques:
* **Weight Conversion:** Transitioning model weights from standard single-precision floating-point format (`FP32`) down to half-precision formats (`FP16` and `BF16`).
* **Impact:** This approach reliably cuts memory bandwidth requirements in half and accelerates hardware-level inference while maintaining the numerical stability required for accurate weather prediction tasks.

### 2. Application-Specific Transformer Pruning
Weather prediction models often rely on stacked local-global attention layers to capture both regional micro-climates and broad synoptic-scale patterns. We exploit structural redundancies within these layers to compress the network:
* **Similarity Scoring:** The system calculates feature representation similarity metrics between sequential transformer layer units (specifically targeting the local-global attention mechanisms).
* **Threshold-Based Pruning:** Layers that exhibit a similarity score above a predefined threshold are dynamically pruned from the network.
* **Impact:** This reduces the overall depth of the model, bypassing redundant transformations and significantly speeding up forward passes without losing the model's core predictive capabilities.

## Setup and Base Model

For the base model setup, environment configuration, and initial weights, this project builds upon the Prithvi WxC foundation model. If you need to set this up from scratch, please refer to their official [Prithvi WxC GitHub repository](https://github.com/NASA-IMPACT/Prithvi-WxC). 

Once the base environment is configured, you can apply the custom quantization and pruning scripts located in the codebase.
