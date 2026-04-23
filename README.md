# ACL Injury Risk Prediction: A Multi-Modal Production Pipeline

## Overview
This project provides a robust machine learning framework to predict ACL injury risk in collegiate athletes. By synthesizing multi-modal sensor data and workload metrics, the pipeline identifies high-risk "Cutting Signatures" and provides actionable load-management protocols for coaching staff.



## Getting Started

### 1. Data Requirements
To run this pipeline, you must first source the necessary datasets. Please download the following files to your local directory:

* **[Athlete Injury and Performance Dataset](https://www.kaggle.com/datasets/ziya07/athlete-injury-and-performance-dataset)**
* **[Multimodal Sports Injury Prediction Dataset](https://www.kaggle.com/datasets/anjalibhegam/multimodal-sports-injury-prediction-dataset)**
* **[University Football Injury Prediction Dataset](https://www.kaggle.com/datasets/yuanchunhong/university-football-injury-prediction-dataset)**

*Ensure these files are placed in the system before executing the script.*

### 2. Execution
To generate the experimental results and validation tables, run the novelty experiment script:
```bash
python3 acl_novelty_experiment.py
```

## Methodology & Novelty
This research builds upon the predictive frameworks established by Jauhiainen (2022), Taborri (2021) and Guo (2025). Our primary novelty lies in Isotonic Probability Calibration, which mitigates the "overconfidence" artifacts common in sports injury modeling, providing coaches with a reliable, clinical-grade tool for load management.

## Project Structure
* /data : Local storage for source CSV files.
* /scripts : Core pipeline and novel experiments.
* /reports : Academic research documentation.
* /visualisations : Generated calibration plots and SHAP feature importance charts.
