# Decision Tree Model - Loan Campaign Analysis

A machine learning project using Decision Tree algorithms to predict customer response to personal loan campaigns and optimize marketing strategies for financial institutions.

## Project Overview

This project implements a Decision Tree classification model to predict whether a bank customer will accept a personal loan offer. The model helps banks identify potential customers for targeted marketing campaigns, improving conversion rates and reducing marketing costs.

## Business Problem

A financial institution wants to optimize its personal loan marketing campaign by targeting customers most likely to accept loan offers. The bank needs to:
- Identify key customer characteristics that predict loan acceptance
- Build a predictive model to score potential customers
- Understand decision rules for customer segmentation
- Reduce marketing costs by focusing on high-potential customers

## Dataset

**File:** `Loan_Modelling.csv`

The dataset contains customer information including:
- Demographics (age, income, family size, education)
- Banking relationship (experience with bank, account types)
- Financial products (credit card, securities account, CD account)
- Previous campaign response
- Target variable: Personal loan acceptance (yes/no)

## Technologies Used

- **Python 3.x**
- **Libraries:**
  - pandas - Data manipulation and analysis
  - numpy - Numerical computations
  - scikit-learn - Machine learning algorithms and model evaluation
  - matplotlib - Data visualization
  - seaborn - Statistical visualization
  - jupyter notebook - Interactive development environment

## Project Structure

```
Decision-tree-model-loan-campaign/
│
├── AIML_ML_Project_full_code_notebook.ipynb  # Main Jupyter notebook with complete analysis
├── Loan_Modelling.csv                         # Customer dataset
├── README.md                                   # Project documentation
└── .gitignore                                  # Git ignore file
```

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Project

1. Clone the repository:
```bash
git clone https://github.com/BhawnaSahnii/Decision-tree-model-loan-campaign.git
cd Decision-tree-model-loan-campaign
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open `AIML_ML_Project_full_code_notebook.ipynb` and run all cells

## Methodology

1. **Data Preprocessing**
   - Data cleaning and handling missing values
   - Exploratory Data Analysis (EDA)
   - Feature engineering and transformation
   - Encoding categorical variables
   - Feature scaling

2. **Model Development**
   - Train-test split
   - Multiple Decision Tree implementations:
     - Decision Tree with sklearn default parameters
     - Decision Tree with Pre-Pruning
     - Decision Tree with Post-Pruning
   - Hyperparameter tuning (max_depth, min_samples_split, criterion)
   - Handling class imbalance

3. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix analysis
   - ROC-AUC Curve
   - Feature importance ranking
   - Decision tree visualization
   - Performance comparison across pruning techniques

4. **Business Insights**
   - Customer segmentation rules
   - Key predictors of loan acceptance
   - Actionable marketing recommendations

## Key Features

- **Interpretable Model:** Decision trees provide clear, understandable rules for loan acceptance prediction
- **Pruning Techniques:** Comparison of different decision tree variants (default, pre-pruning, post-pruning) to optimize model complexity and prevent overfitting
- **Feature Importance:** Identification of most influential customer characteristics
- **Visual Decision Rules:** Tree visualization for easy interpretation by business stakeholders
- **Performance Metrics:** Comprehensive evaluation of model accuracy and business impact
- **Comparative Analysis:** Test set performance comparison across different pruning strategies

## Learning Outcomes

- Understanding Decision Tree algorithms and their interpretability
- Implementing and comparing pre-pruning and post-pruning techniques
- Managing model complexity and preventing overfitting
- Implementing classification models for imbalanced datasets
- Hyperparameter tuning for optimal model performance
- Feature selection and engineering techniques
- Translating model outputs into business recommendations
- Model evaluation and validation strategies

## Business Impact

- Improved targeting of potential loan customers
- Reduced marketing costs through focused campaigns
- Higher conversion rates for personal loan offers
- Better understanding of customer behavior and preferences
- Data-driven decision making for marketing strategies

## License

This project is open source and available for educational purposes.

## Acknowledgments

This project was completed as part of the **Applied Machine Learning** course offered by **Great Learning** in collaboration with **The University of Texas at Austin**.

---

If you found this project helpful, please consider giving it a star!
