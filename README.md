# Credit Scoring Model

A machine-learning project that predicts whether a loan application is likely to be approved from an applicant's financial, demographic, and loan-related information.

The work is contained in `credit_scoring_model.ipynb`. It prepares the data, creates useful financial features, trains two classification models, compares their performance, and evaluates the stronger model.

## Project objective

The goal is to classify each application as **Approved** or **Not Approved**. This is a binary classification problem where the target column is `Loan_Approved`.

## Dataset

The notebook expects a CSV file named `data.csv` in the same folder as the notebook.

```text
Task1/
├── credit_scoring_model.ipynb
├── README.md
└── data.csv
```

The dataset contains applicant and loan information, including:

- Applicant and coapplicant income
- Employment status, education level, gender, marital status, and dependents
- Credit score, existing loans, debt-to-income ratio, and savings
- Collateral value, requested loan amount, loan term, and loan purpose
- Property area and employer category
- Loan approval outcome

The identifier column, `Applicant_ID`, is removed because it does not describe an applicant's ability to repay a loan. Rows without a `Loan_Approved` value are removed because they have no outcome for the model to learn from.

## Data preparation

The notebook prepares the data inside a scikit-learn pipeline:

- Numerical missing values are filled with the median.
- Categorical missing values are filled with the most frequent value.
- Categorical variables are converted with one-hot encoding.
- Numerical variables are standardized.
- The data is split into 80% training data and 20% test data using stratified sampling.

## Feature engineering

Four additional financial features are created:

| Feature | Description |
| --- | --- |
| `Total_Income` | Applicant income plus coapplicant income |
| `Loan_to_Income` | Requested loan amount relative to total income |
| `Collateral_Coverage` | Collateral value relative to requested loan amount |
| `Income_per_Dependent` | Total income divided by the number of dependents plus one |

These features help the models assess affordability and the financial coverage of a loan.

## Models

Two supervised classification models are trained:

| Model | Why it is included |
| --- | --- |
| Logistic Regression | A clear baseline model for binary classification |
| Random Forest | Can learn more complex relationships and provide feature-importance values |

Both models use balanced class weights to reduce the impact of an uneven number of approved and non-approved applications. The Random Forest uses 400 trees and a minimum leaf size of 3.

## Model evaluation

The notebook compares the models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The model with the highest ROC-AUC is selected as the best model. The notebook then produces:

- A classification report
- A confusion matrix
- An ROC curve
- A chart of the top 15 important features when Random Forest is selected

## Results

Using the notebook's 80/20 stratified train-test split with a random seed of 42, Random Forest achieved the strongest performance and was selected as the final model.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.957895 | 0.959534 | 0.957895 | 0.958245 | 0.980128 |
| Logistic Regression | 0.847368 | 0.859298 | 0.847368 | 0.850405 | 0.927692 |

The Random Forest model achieved approximately **95.8% accuracy** and a **0.980 ROC-AUC**, outperforming Logistic Regression across every reported metric.

## How to run

1. Clone or download this project.
2. Put `data.csv` in the same folder as `credit_scoring_model.ipynb`.
3. Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

4. Open the notebook in Jupyter Notebook, JupyterLab, or VS Code.
5. Run the cells from top to bottom.

## Notes and limitations

This project is designed for learning and demonstration. A model trained on historical loan data can inherit bias from that data. Before using a model like this in a real lending workflow, it should be tested for fairness, data quality, privacy protection, explainability, and compliance with applicable lending regulations.

Predictions should support human decision-making, not replace it.
