# 🏀🏈 Court & Gridiron Outcome Lab

A machine learning project to **predict NBA and NFL game outcomes** using player and team statistics combined with contextual game features.  
Built with **Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, and XGBoost**.

---

## ✨ Features
- **Synthetic Data Generators**  
  Create realistic NBA and NFL datasets with Elo ratings, player metrics, injuries, travel, rest days, weather, and more.

- **Exploratory Data Analysis (EDA)**  
  Visualize distributions, correlations, and feature importance using Seaborn & Matplotlib.

- **Machine Learning Models**  
  Train and evaluate:
  - Logistic Regression
  - Random Forest
  - XGBoost

- **Model Optimization**  
  Hyperparameter tuning & cross-validation to improve F1-score and precision-recall trade-offs.

---

## 📦 Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/Court-Gridiron-Outcome-Lab.git
cd Court-Gridiron-Outcome-Lab
python -m pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Generate synthetic datasets
**NBA**
```bash
python src/data.py --generate 4000 --seed 13 --sport nba --out data/simulated_games_nba.csv
```

**NFL**
```bash
python src/data.py --generate 4000 --seed 13 --sport nfl --out data/simulated_games_nfl.csv
```

---

### 2. Run EDA
This produces plots in the `plots/` directory.

```bash
python src/eda.py --input data/simulated_games_nba.csv
python src/eda.py --input data/simulated_games_nfl.csv
```

---

### 3. Train Models
Train Logistic Regression, Random Forest, and XGBoost:

```bash
python src/train.py --input data/simulated_games_nba.csv --models lr rf xgb --cv 5 --scoring f1 --n_iter 25
python src/train.py --input data/simulated_games_nfl.csv --models lr rf xgb --cv 5 --scoring f1 --n_iter 25
```

Reports and metrics will be saved in the `reports/` folder.  
Performance plots (ROC, PR curves, confusion matrices) will be saved in the `plots/` folder.

---

## 📊 Project Structure
```
Court-Gridiron-Outcome-Lab/
│
├── data/                 # Generated NBA/NFL datasets
├── plots/                # Visualization outputs
├── reports/              # Model evaluation reports
│
├── src/
│   ├── data.py           # Data generation + validation
│   ├── eda.py            # Exploratory Data Analysis
│   ├── train.py          # Model training & tuning
│
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🛠 Tech Stack
- **Python 3.9+**
- **NumPy** & **Pandas** → preprocessing & feature engineering  
- **scikit-learn** → ML models & cross-validation  
- **XGBoost** → gradient boosting  
- **Matplotlib** & **Seaborn** → EDA & visualization  

---

## 📈 Example Outputs
- Correlation heatmaps of features vs. outcomes  
- ROC and Precision-Recall curves  
- Confusion matrices  
- Classification reports with F1, Precision, Recall  

---

## 🤝 Contributing
Feel free to fork the repo and submit pull requests for new features, improvements, or bug fixes.

---

## 📜 License
This project is licensed under the MIT License.
