# Intermittent Demand Forecasting with Data Augmentation

Research project investigating whether synthetic data augmentation can improve **intermittent-demand time-series forecasting** under limited training data.

The project uses the **Monash Car Parts dataset** and evaluates multiple augmentation strategies with **PatchTST**, alongside classical intermittent-demand forecasting baselines.

## Overview

Intermittent demand is characterised by long periods of zero demand interrupted by irregular demand spikes. This makes forecasting particularly challenging, especially when only a small amount of historical data is available.

This project investigates the following question:

> **Can structure-aware synthetic data augmentation improve forecasting performance for intermittent demand when training data is scarce?**

The experiments compare a real-data-only PatchTST baseline against several augmentation strategies:

* Gaussian **Jittering**
* **Poisson Bootstrap**
* **Raw-space Poisson Bootstrap**
* **Magnitude-only Bootstrap**
* **TimeGAN**

Performance is evaluated using **MASE (Mean Absolute Scaled Error)**, both globally and across demand regimes.

## Dataset

The experiments use the **Monash Car Parts** time-series dataset, loaded through the `autogluon/chronos_datasets` mirror.

The notebook performs several preprocessing and quality-control steps:

1. Load the Car Parts time series.
2. Remove unsuitable series based on length, zero ratio, and number of non-zero observations.
3. Split each series into:

   * **39 training timesteps**
   * **12 held-out test timesteps**
4. Apply `log(1+x)` transformation.
5. Standardise using statistics calculated from the training data only.

Three training-data scarcity regimes are defined:

| Regime   | Training timesteps |
| -------- | -----------------: |
| Extreme  |                 15 |
| Moderate |                 20 |
| Full     |                 39 |

## Forecasting Models

### Classical baselines

The project evaluates several statistical forecasting approaches using `StatsForecast`:

* Naive
* Seasonal Naive
* Croston Optimized
* ADIDA

### Neural forecasting

The primary neural forecasting model is **PatchTST**, implemented through `NeuralForecast`.

The main configuration includes:

* Forecast horizon: 12 months
* Input size: 24
* Patch length: 6
* Stride: 3
* Attention heads: 4
* Encoder layers: 3
* Hidden size: 128
* Linear hidden size: 256
* Dropout: 0.1
* Maximum training steps: 500

## Data Augmentation Methods

### 1. Jittering

Gaussian noise is added to the original training series:

`x' = x + N(0, noise_std × σ)`

This provides a simple baseline for synthetic augmentation.

### 2. Poisson Bootstrap

The intermittent structure is explicitly modelled by separating:

* non-zero demand magnitudes
* inter-arrival intervals between demand events

Both components are resampled to reconstruct synthetic demand series.

### 3. Raw-space Poisson Bootstrap

Poisson Bootstrap is performed in the **original demand space**, followed by the same `log(1+x)` and standardisation pipeline used for the real data.

This is intended to avoid distortions introduced by performing the bootstrap after scaling.

### 4. Magnitude-only Bootstrap

Demand spike positions and therefore the zero-inflation pattern are preserved, while non-zero spike magnitudes are resampled.

### 5. TimeGAN

A custom recurrent TimeGAN-style architecture is trained to learn the temporal distribution of the demand series.

The implementation contains:

* Embedder
* Recovery network
* Generator
* Supervisor
* Discriminator

GRU layers and Layer Normalisation are used, with separate autoencoder, supervisor, and joint adversarial training phases.

## Evaluation

Forecasts are evaluated using **MASE**.

Evaluation is performed at multiple levels:

### Global performance

Overall median MASE across the test series.

### Demand-regime performance

The test observations are decomposed into:

* **Zero demand**
* **Low demand**
* **High demand**

The high-demand threshold is defined using the 75th percentile of non-zero raw test observations.

This decomposition is important because an augmentation method may improve overall forecasting accuracy while performing poorly on demand spikes.

## Key Results

The notebook reports the following full-regime comparison:

| Method              | Global MASE |       Zero |        Low |       High |
| ------------------- | ----------: | ---------: | ---------: | ---------: |
| Real-Only           |      0.9376 |     0.4079 |     1.9427 |     2.8475 |
| Jitter              |      0.9502 |     0.4324 |     1.9349 |     2.8399 |
| PoissonBS-Sc        |      0.9510 |     0.4131 |     2.0155 |     2.9122 |
| PoissonBS-Raw       |      0.9362 |     0.3897 |     1.9770 |     2.8858 |
| Magnitude Bootstrap |      0.9402 |     0.4048 |     1.9551 |     2.8831 |
| **TimeGAN**         |  **0.9044** | **0.3158** | **1.9652** | **2.8170** |

The notebook therefore reports **TimeGAN as the strongest method on global MASE and zero-demand MASE** among the evaluated augmentation methods.

An important observation is that improvements are not uniform across regimes. For example, TimeGAN improves the global and zero-demand metrics relative to the real-only baseline, while the low-demand metric remains slightly higher and the high-demand metric improves only modestly.

## Project Structure

```text
intermittent-demand-forecasting-augmentation/
│
├── notebooks/
│   └── Monash_augmentation_1.ipynb
│
├── results/
│   ├── checkpoint1_real_only_full_regime.csv
│   ├── checkpoint1_summary.json
│   ├── checkpoint1c_regime_decomposition.csv
│   ├── checkpoint1c_summary.csv
│   ├── checkpoint2_jitter_global.csv
│   ├── checkpoint2_jitter_decomp.csv
│   ├── checkpoint3c_magbs_global.csv
│   ├── checkpoint3c_magbs_decomp.csv
│   ├── checkpoint4_timegan_global.csv
│   ├── checkpoint4_timegan_decomp.csv
│   ├── checkpoint4_full_comparison.csv
│   ├── tier2_bootstrap_summary.csv
│   ├── full_tier_summary.csv
│   └── full_tier_delta.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Create a Python environment and install the required dependencies:

```bash
pip install torch torchvision
pip install neuralforecast
pip install statsforecast
pip install lightgbm
pip install datasets
pip install matplotlib seaborn scipy scikit-learn
pip install tensorflow==2.15.0
pip install git+https://github.com/ydataai/ydata-synthetic.git
```

The notebook also automatically installs the main dependencies when executed.

## Running the Project

Open the notebook:

```text
notebooks/Monash_augmentation_1.ipynb
```

Then execute the cells sequentially.

The experiments are computationally intensive, particularly the PatchTST and TimeGAN training stages. GPU acceleration is supported and the notebook automatically selects CUDA when available.

## Reproducibility

A fixed random seed is used:

```python
SEED = 42
```

The experiments also explicitly control the training/test horizon and scarcity regimes.

## Research Direction

The current experiments suggest that **augmentation strategy matters substantially for intermittent demand**.

Simple perturbation methods such as jittering do not necessarily improve forecasting performance. Structure-aware approaches perform differently depending on whether synthetic data preserves:

* zero-inflation
* demand-event timing
* spike magnitude
* temporal dependencies

The TimeGAN results provide evidence that learning the underlying temporal distribution may be a promising direction for improving forecasting when historical observations are limited.

Future work could investigate:

* More rigorous repeated-seed experiments
* Additional intermittent-demand datasets
* Alternative generative time-series models
* Conditional generation based on demand characteristics
* Better preservation of extreme demand spikes
* Statistical significance testing across augmentation methods
* Ablations across different synthetic-to-real data ratios
* Evaluation under the extreme and moderate scarcity regimes

## Author

Research project on **time-series forecasting, intermittent demand, synthetic data generation, and data augmentation**.

## License

Add an appropriate license before publishing the repository.
