# 🎓 Student Performance Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%2C%20Scikit--learn%2C%20Seaborn-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-green.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

A machine learning project to predict student academic performance and identify key factors influencing success, helping educational institutions provide proactive support.

---

### 🚀 Project Overview

This project addresses the challenge of identifying at-risk students by building predictive models based on demographic, social, and academic data. The goal is to provide educational institutions with a data-driven tool for early intervention. The project involves two main tasks:
1.  **Regression:** Predicting a student's final numeric grade.
2.  **Classification:** Predicting whether a student will pass or fail.

![Animated GIF of a neural network learning](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbDBkMGwzY3JzY2Z4cnA4a3R2MWI0bHh5bnd2cjN6cnJ2cGRoc2F3dCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/L2pr3M2Vib16w/giphy.gif)

### ✨ Key Features

- **Data Exploration (EDA):** In-depth analysis of student data to uncover initial trends and correlations.
- **Data Preprocessing:** A complete pipeline for cleaning data, encoding categorical variables, and scaling features.
- **Dual-Task Modeling:** Implements both regression and classification models to provide a comprehensive performance analysis.
- **Performance Evaluation:** Uses a wide range of metrics (R², RMSE, Accuracy, Precision, Recall, F1-Score) for robust model assessment.
- **Feature Importance Analysis:** Identifies the key drivers of academic success from the dataset.

### ⚙️ Methodology

The project follows a standard machine learning workflow:

1.  **Data Understanding:** The [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance) was used. Initial analysis was performed to understand its structure, quality, and statistical properties.
2.  **Exploratory Data Analysis (EDA):** Visualizations such as histograms and a correlation heatmap were created to identify relationships between variables, especially their impact on the final grade (`G3`).
3.  **Data Preprocessing:**
    - A binary `pass_fail` feature was engineered from the `G3` grade.
    - Categorical features were converted to a numerical format using one-hot encoding.
    - All features were scaled using `StandardScaler` to prepare the data for modeling.
4.  **Model Building:**
    - **Regression Task:** A Multiple Linear Regression model was trained to predict the final grade.
    - **Classification Task:** Logistic Regression and Decision Tree models were trained to predict the pass/fail outcome.
5.  **Model Evaluation:** The models were evaluated on an unseen test set (20% of the data) to measure their real-world performance.

### 📊 Results & Key Insights

The models demonstrated strong predictive capabilities:

| Model / Task | Metric | Score |
| :--- | :--- | :--- |
| **Linear Regression** | R² Score | **0.72** |
| **Classification Models** | Accuracy | **90%** |
| (Logistic & Decision Tree) | F1-Score | **0.92** |

**Key Insight:** The feature importance analysis revealed that a student's **second-period grade (`G2`)** is overwhelmingly the most significant predictor of their final academic outcome, accounting for over 70% of the decision-making power in the model. Other important factors include **parental education (`Medu`)**, student **absences**, and social habits.

### 🔧 How to Run This Project

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourRepositoryName.git](https://github.com/YourUsername/YourRepositoryName.git)
    cd YourRepositoryName
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    Create a `requirements.txt` file with the following content:
    ```
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    jupyter
    ```
    Then, run the installation command:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset:**
    - Download the `student-mat.csv` file from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/student+performance).
    - Place the `student-mat.csv` file in the root directory of the project.

5.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Open the `.ipynb` notebook file and run the cells.

### 🛠️ Technologies Used

| Technology | Description |
| :--- | :--- |
| **Python** | Core programming language for the project. |
| **Pandas** | Data manipulation and analysis library. |
| **NumPy** | For numerical operations and array handling. |
| **Scikit-learn** | For building and evaluating machine learning models. |
| **Matplotlib & Seaborn** | For data visualization and creating plots. |
| **Jupyter Notebook / Colab**| For interactive development and documentation. |

---
