

# 🧠 Product Category Classification using Machine Learning

This project implements a multi-class classification system to automatically assign product titles to appropriate categories using machine learning techniques. It is designed as part of a college-level ML course and demonstrates end-to-end development of a predictive pipeline.

## 📂 Project Structure

```
ProductClassification/
├── data/                   # Original dataset
├── src/                    # Source code (EDA, preprocessing, training, evaluation)
├── models/                 # Saved trained models (.joblib)
├── reports/                # Run outputs and generated logs
├── figures/                # Visualization outputs (charts, confusion matrix)
├── ReadME.md               # Project overview
└── environment.yml         # Conda environment file
```

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets)
- **Size**: 35,311 product titles
- **Fields**: Product ID, Title, Merchant ID, Cluster ID, Category ID, etc.

The task involves classifying each product into one of several categories such as *Mobile Phones*, *Dishwashers*, *Fridges*, and more.

## ⚙️ Features

- Data preprocessing (cleaning, text normalization, TF-IDF)
- Label encoding of categories
- Train-test split (stratified)
- Models: Logistic Regression (with GridSearchCV), Neural Network (MLPClassifier)
- Performance evaluation (accuracy, precision, recall, F1, confusion matrix)
- Output logging and auto-reporting
- Rich EDA visualizations

## 🚀 How to Run

```bash
# Step 1: Clone the repo
git clone git@github.com:HarshBhadania05/Product-Classification.git
cd ProductClassification

# Step 2: Set up the environment
conda env create -f environment.yml
conda activate productClassification

# Step 3: Run the pipeline
python main.py
```

All outputs (logs, plots, models) will be saved to `reports/` and `figures/`.

## 📘 Report

A full report in PDF format is available in the `reports/` folder and includes:
- Project motivation
- Dataset exploration
- Preprocessing pipeline
- Model details and justification
- Evaluation, visualizations, and interpretation
- References and acknowledgements

## 📎 Acknowledgements

- Data: UCI ML Repository
- Libraries: Scikit-learn, Pandas, Matplotlib, Seaborn
- Tools: VS Code, GitHub, Conda
- Assisted by OpenAI’s ChatGPT for debugging

## 🔗 License

This project is for educational use only.

## 📬 Contact

**Harsh Bhadania**  
University of North Carolina at Charlotte  
hbhadani@uncc.edu