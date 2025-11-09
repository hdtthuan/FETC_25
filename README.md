# 🔋 Forecasting Urban Energy for Net-Zero Smart Cities  
### A Hybrid Attention-based Deep Learning Approach  

**Authors:**  
- **Minh Anh Hoang** – Swinburne University Vietnam · FPT University, HCM City, Vietnam  
- **Thuan Do Thanh Hoang** – FPT University, HCM City, Vietnam  
- **Tuan Phu Phan** – FPT University, HCM City, Vietnam  
- **Khuong Nguyen-Vinh** – RMIT University, HCM City, Vietnam  

📘 *Published in the Vietnam Journal of Computer Science (World Scientific Publishing)*  
📄 [View Official Publication (FETC)](https://science.fpt.edu.vn/FETC/AcceptedPaper/PaperDetail?id=9a3f5017-99c5-4e68-a6ee-08ddcd88f2d6)  
🗓️ **Accepted:** November 2025  
📧 Corresponding Author: minhha10@fe.edu.vn  

---

## 🌍 Overview

This repository contains the code, documentation, and final paper for our research **“Forecasting Urban Energy for Net-Zero Smart Cities: A Hybrid Attention-based Approach.”**  
The project proposes **two hybrid deep learning architectures** that integrate **Convolutional Neural Networks (CNNs)**, **Long Short-Term Memory (LSTM)**, and **Attention mechanisms** to enhance **short-term and day-ahead energy forecasting** in smart cities.

Our models are evaluated on the **ENTSO-E European Energy Dataset (2019–2025)**, achieving state-of-the-art results across multiple forecasting horizons — demonstrating the critical role of hybrid AI models in **sustainable energy management** and **net-zero urban planning**.

---

## 🧠 Research Highlights

| Model | Task | RMSE (MW) | MAE (MW) | MAPE (%) |
|:------|:-----|:----------:|:---------:|:---------:|
| **CNN-LSTM-Att.v2** | Single-Step Forecast | **106.96** | **80.22** | **1.07** |
| **CNN-LSTM-Att.v1** | Multi-Step (Day-Ahead) Forecast | **438.11** | **315.32** | **4.18** |
| XGBoost | Multi-Step | 500.59 | 346.36 | 4.61 |
| GRU / LSTM | Multi-Step | 612.29 | 448.73 | 5.98 |

✅ *CNN-LSTM-Att.v2* excels in real-time short-term forecasting (lightweight, efficient).  
✅ *CNN-LSTM-Att.v1* delivers superior day-ahead prediction accuracy for centralized systems.

---

## ⚙️ Methodology

### 🧩 Model Architecture

Each model integrates:
1. **CNN Encoder** – Extracts short-term temporal features via 1D convolutions.  
2. **Stacked LSTM Layers** – Captures long-range sequential dependencies.  
3. **Attention Mechanism** – Dynamically weighs relevant time steps to improve interpretability and prediction quality.  
4. **Dense Output Layer** – Produces either single-step or 24-step (day-ahead) predictions.

### 🔬 Formulation

\[
M(X) = \phi_{out} \circ \phi_{att} \circ \phi_{LSTM} \circ \phi_{CNN}(X)
\]

- Loss Function: Mean Squared Error (MSE)  
- Optimizer: Adam (lr = 1e-4)  
- Framework: TensorFlow / Keras  
- Environment: Google Colab (Tesla T4 GPU)

---

## 📊 Dataset

**Dataset:** ENTSO-E (European Network of Transmission System Operators for Electricity)  
**Period:** 2019 – 2025  
**Features:**
- Hourly electricity load (MW)
- Temporal features: hour, weekday, month, year  
**Preprocessing:**
- Missing values removed  
- Min-Max normalization  
- Chronological split → Train (2019–2022), Validation (2023), Test (2024–2025)

---

## 📈 Results Visualization

| Forecast Type | Description |
|----------------|--------------|
| **Single-Step Prediction** | CNN-LSTM-Att.v2 achieves <110 MW RMSE and <1.1% MAPE. |
| **Multi-Step Day-Ahead** | CNN-LSTM-Att.v1 maintains <4.2% MAPE over 24-hour horizon. |
| **Comparison** | Hybrid models outperform GRU, LSTM, Dense, and ensemble baselines. |

Figures in the paper illustrate:
- Predicted vs. Actual Load Curves  
- Multi-Step Forecast Comparisons across Models  
- Energy Distribution by Country (France, Luxembourg, Austria)

---

## 🧱 Repository Structure

```

Forecasting_Urban_Energy/
├── README.md
├── paper/
│   └── Forecasting_Urban_Energy_for_Net_Zero_Smart_Cities.pdf
├── src/
│   ├── model_cnn_lstm_att_v1.py
│   ├── model_cnn_lstm_att_v2.py
│   ├── baseline_lstm_gru.py
│   ├── xgboost_baseline.py
│   └── utils/
│       ├── data_loader.py
│       └── preprocess.py
└── notebooks/
└── experiments.ipynb

````

---

## 🧪 Reproduction Guide

### Setup
```bash
git clone https://github.com/<your-username>/Forecasting_Urban_Energy.git
cd Forecasting_Urban_Energy
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
````

### Training

```bash
python src/model_cnn_lstm_att_v2.py --train --epochs 100
```

### Evaluation

```bash
python src/evaluate.py --model cnn_lstm_att_v2 --task multi_step
```

---

## 🔍 Key Insights

* **Hybridization Matters:** Combining CNNs, LSTMs, and Attention yields superior generalization.
* **Model Depth vs. Efficiency:** Lightweight architectures (v2) are ideal for edge and real-time smart-meter applications.
* **Explainability Challenge:** Future research should integrate SHAP, LIME, or Integrated Gradients to validate attention interpretability.
* **Scalability:** Federated and blockchain-integrated forecasting systems can ensure decentralized, privacy-preserving smart-grid management.

---

## 📚 Citation

If you reference this work, please cite:

```bibtex
@article{hoang2025hybridenergy,
  title     = {Forecasting Urban Energy for Net-Zero Smart Cities: A Hybrid Attention-Based Approach},
  author    = {Minh Anh Hoang and Thuan Do Thanh Hoang and Tuan Phu Phan and Khuong Nguyen-Vinh},
  journal   = {Vietnam Journal of Computer Science},
  publisher = {World Scientific Publishing},
  year      = {2025},
  url       = {https://science.fpt.edu.vn/FETC/AcceptedPaper/PaperDetail?id=9a3f5017-99c5-4e68-a6ee-08ddcd88f2d6}
}
```

---

## 🧭 Acknowledgement

This work was supported by **Swinburne Vietnam**, **FPT University**, and **RMIT University Vietnam**.
Special thanks to our advisors and collaborators for guidance throughout the research.

---

## 🪄 License

© 2025 Authors. All rights reserved.
This repository is provided for **academic and research purposes**.
Please contact the corresponding author for collaboration or reuse permissions.

---

```

---

Would you like me to extend this README with a **“Figure Gallery”** section that embeds key result plots (Predicted vs Actual, Multi-Step Forecast Comparison, Architecture Diagram) in Markdown format — suitable for a GitHub landing page?
```
