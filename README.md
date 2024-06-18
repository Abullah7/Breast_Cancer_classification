# Breast Cancer Detection using Machine Learning

This repository contains a machine learning project for detecting breast cancer using a dataset from Kaggle. The project demonstrates data preprocessing, model training, evaluation, and prediction. The dataset used for this project is the Breast Cancer Wisconsin (Diagnostic) Data Set.

## Dataset

The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets) and contains features computed from digitized images of fine needle aspirates (FNA) of breast mass. The dataset includes 30 feature columns and one target column indicating whether the tumor is malignant or benign.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/breast-cancer-detection.git
    cd breast-cancer-detection
    ```

2. **Install dependencies**:

    Ensure you have Python installed, then install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main notebook for this project is `model_1.ipynb`, which includes the following steps:

1. **Loading the dataset**: Importing the dataset and displaying the initial data.
2. **Data Preprocessing**: Cleaning the data, handling missing values, and encoding categorical variables.
3. **Exploratory Data Analysis (EDA)**: Visualizing data distributions, correlations, and relationships between features.
4. **Model Training**: Training various machine learning models to find the best classifier.
5. **Model Evaluation**: Evaluating model performance using metrics like accuracy, precision, recall, and F1-score.
6. **Prediction**: Making predictions on new data and validating model performance.

To run the notebook:

```bash
jupyter notebook model_1.ipynb
```

## Results

The project evaluates multiple machine learning models including Logistic Regression, Decision Trees, Random Forests, and Support Vector Machines (SVM). The best-performing model is selected based on evaluation metrics.

### Key Metrics

- **Accuracy**: Measure of the model's overall correctness.
- **Precision**: Measure of the correctness of positive predictions.
- **Recall**: Measure of the model's ability to identify positive instances.
- **F1-Score**: Harmonic mean of precision and recall, providing a single metric to evaluate the model's performance.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you have suggestions or bug reports.

## License

This project is licensed under the MIT License.

## Acknowledgements

- The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).
- Special thanks to the contributors of the dataset and the open-source community.

---

Feel free to reach out with any questions or suggestions!


