# Machine Learning for Bank Loan Risk Assessment

Predicts financial/loan risk from applicant data, comparing classical ML, ensemble methods and neural networks across several data-balancing and feature-reduction strategies.

## Pipeline

1. **Preprocessing** – missing-value handling, dropping unneeded features, engineering new ones, normalisation and categorical encoding.
2. **EDA** – distribution and correlation plots with Matplotlib/Seaborn.
3. **Class balancing** – SMOTE (the original notebook also tries ADASYN, SMOTEENN and SMOTETomek).
4. **Feature ranking, selection and reduction** – Random Forest importances, SelectKBest, RFE, PCA, t-SNE.
5. **Models** – Logistic Regression, Decision Tree, Random Forest, SVM, k-NN, Naive Bayes, Gradient Boosting, XGBoost, Bagging, Stacking, and Keras/MLP neural networks.
6. **Evaluation** – accuracy, precision, recall and F1 compared across the balanced, feature-selected and feature-reduced datasets.

## Repository contents

| File | Description |
|------|-------------|
| `ML_FINAL_PROJECT.ipynb` | Final, cleaned-up notebook |
| `ML_FINAL_PROJECT_Original.ipynb` | Original notebook including extra experiments |
| `ML_endsem.pptx` | Presentation slides |

## Setup

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost tensorflow matplotlib seaborn
```

The notebooks read `financial_risk_assessment.csv`, which is not included in this repository. Add it to the project folder (or update the path in the notebook) and run top to bottom.

## Tech stack

Python, pandas, scikit-learn, imbalanced-learn, XGBoost, TensorFlow/Keras

## Author

Sai Swaroop Talasu – [@saiswaroop25-pixel](https://github.com/saiswaroop25-pixel)
