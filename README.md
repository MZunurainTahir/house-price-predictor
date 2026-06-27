# 🏠 House Price Predictor

A from-scratch implementation of **Linear Regression** (Normal Equation, Batch GD, Stochastic GD, Mini-batch GD) benchmarked against **scikit-learn**, applied to the real Ames Housing dataset to predict residential sale prices.

This project is part of a 10-project ML portfolio series, each pairing a core ML algorithm implemented from mathematical first principles with a real-world dataset and an sklearn comparison.

## 📌 Project Highlights

- ✅ Linear Regression derived and implemented from scratch (NumPy only) — MSE loss, gradients, **and the closed-form Normal Equation**
- ✅ **Four solution methods implemented and compared:** Normal Equation, Batch GD, SGD, Mini-batch GD
- ✅ All from-scratch metrics (RMSE, MAE, R²) validated against `sklearn.metrics` to machine precision
- ✅ Real Ames Housing dataset (1,460 houses, Kaggle's House Prices competition data)
- ✅ Full EDA: correlation analysis, neighborhood price comparison, distribution + skew analysis
- ✅ 3D regression surface visualization
- ✅ **Honest hyperparameter-tuning narrative** — documents how/why SGD needed different settings to perform competitively, instead of hiding the tuning process

## 📊 Results Summary

| Model | R² | RMSE ($) | MAE ($) | Train Time |
|---|---|---|---|---|
| Linear Regression (Normal Equation, scratch) | 0.8703 | $29,528 | $19,034 | 0.001s |
| Linear Regression (GD, scratch) | **0.8710** | $29,513 | $19,019 | 0.31s |
| Linear Regression (SGD, scratch) | 0.8432 | $31,630 | $22,010 | 1.01s |
| Linear Regression (Mini-batch, scratch) | 0.8657 | $29,128 | $18,996 | 0.79s |
| Linear Regression (sklearn) | 0.8703 | $29,528 | $19,034 | 0.002s |

📄 Full report with all charts and the SGD tuning story: [`reports/report.md`](reports/report.md)

## 🗂️ Project Structure

```
house-price-predictor/
├── data/
│   ├── download_data.py        # reproducible dataset download script
│   ├── ames_housing.csv         # Ames Housing dataset (1,460 houses)
│   └── DATA_DICTIONARY.md       # column definitions
├── src/
│   ├── preprocessing.py         # feature selection, encoding, from-scratch StandardScaler
│   ├── linear_regression_scratch.py  # core from-scratch model (Normal Eq/GD/SGD/Mini-batch)
│   ├── evaluate.py              # from-scratch metrics (RMSE, MAE, R²)
│   ├── train_and_compare.py     # main experiment: from-scratch vs sklearn
│   └── generate_visuals.py      # generates all charts in reports/figures/
├── notebooks/                    # exploratory + walkthrough notebook
├── reports/
│   ├── report.md                # full detailed report with embedded charts
│   ├── model_comparison_results.csv
│   └── figures/                  # all PNG charts (EDA, loss curves, 3D plot, etc.)
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

```bash
git clone https://github.com/<your-username>/house-price-predictor.git
cd house-price-predictor

pip install -r requirements.txt

python data/download_data.py          # downloads the dataset
python src/train_and_compare.py       # trains all 5 models, prints + saves metrics
python src/generate_visuals.py        # regenerates all charts
```

## 🧠 The Math (short version)

Linear Regression models `y_hat = w·x + b`, trained by minimizing Mean Squared Error. Unlike logistic regression, it also has an **exact closed-form solution** (the Normal Equation):

```
w = (Xᵗ X)⁻¹ Xᵗ y
```

This project implements both the closed-form solution and all three gradient-descent variants, and shows when each is the right tool. Full derivation in [`src/linear_regression_scratch.py`](src/linear_regression_scratch.py) and the [report](reports/report.md).

## 📁 Dataset

Ames Housing dataset (Dean De Cock) — 1,460 residential home sales in Ames, Iowa (2006–2010), 79 explanatory features. This project uses an 11-feature numeric subset plus neighborhood, selected by correlation strength. See [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md).

## ⚠️ Disclaimer

This is an educational/portfolio project, not a production real-estate valuation tool.

## 📜 License

MIT
