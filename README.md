
---

# Deep Learning-Based Root Cause Detection of Water Contamination in IoT-Enabled Smart Water Networks

This repository contains the official source code and datasets for the paper: **"Deep Learning-Based Root Cause Detection of Water Contamination in IoT-Enabled Smart Water Networks via Sensor Scheduling and Edge-Cloud Computing"**.

Authored by:
*   Zeinab Mahzoon (s.z.mahzoon@gmail.com)
*   Omid Bushehrian (bushehrian@sutech.ac.ir)
*   Pirooz Shamsinejad (p.shamsinejad@sutech.ac.ir)

Department of Computer Engineering and Information Technology, Shiraz University of Technology, Shiraz, Iran.

## Architecture Overview

The proposed system strategically decouples the problem into two distinct stages:
1.  **Edge Layer (Anomaly Detection):** Lightweight, univariate LSTM models are deployed on edge servers. Each model learns the normal behavior of an individual sensor and flags any significant deviation as an anomaly. This generates simple binary signals instead of transmitting raw, high-volume data.
2.  **Cloud Layer (Source Localization):** A powerful XGBoost classifier in the cloud receives the binary anomaly signals from all edge nodes. It analyzes the network-wide spatio--temporal pattern of these signals to infer the contamination source node.
3.  **Optimization (Sensor Scheduling):** An NSGA-II based multi-objective optimization algorithm generates optimal sensor activation schedules to balance the conflicting goals of maximizing localization accuracy and extending the network's operational lifetime.


## Prerequisites

The code is written in Python 3. The main libraries required to run the experiments are listed below:

*   TensorFlow (Keras)
*   XGBoost
*   Pandas
*   NumPy
*   Scikit-learn
*   Matplotlib

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/zeinabmahzon/RootCauseDetection-IoT-Edge-CloudComputing.git
    cd RootCauseDetection-IoT-Edge-CloudComputing
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install tensorflow xgboost pandas numpy scikit-learn matplotlib
    ```

## Datasets

The datasets used in this study were generated using the **EPANET** simulation toolkit. We provide the generated datasets in the `Datasets/` directory. The datasets model networks of three different sizes:
*   **DS-S:** Small (25 Nodes)
*   **DS-M:** Medium (50 Nodes)
*   **DS-L:** Large (100 Nodes)


