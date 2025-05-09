ML Stock Classifier – Buy/Sell Predictor”

This project builds a supervised ML model to predict whether AAPL stock will go **UP or DOWN** the next day using technical indicators.

---

## 🧠 Features

- `Sma_5` – 5-day Simple Moving Average
- `Sma_10` – 10-day SMA
- `Vol_10` – 10-day rolling volatility
- `Signal` – 1 (BUY) if next day return > 0, else 0

---

## 📊 Model Performance

- Classifier: **Random Forest (n=100)**
- Accuracy: **66.7%**
- Evaluation: **Confusion Matrix + Feature Importance**

![Feature Importance](importance_plot.png)

---

## 💾 Files

- `ml_stock_classifier.py` – training & evaluation script
- `aapl.csv` – historical data (2024)
- `importance_plot.png` – visual explanation
- `README.md` – this file

---

## 🧾 Author

Shiva Sai | Quant + FinTech Track | Pre-WashU MSF 2025
