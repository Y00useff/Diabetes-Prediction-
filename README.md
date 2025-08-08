# Diabetes Prediction using Machine Learning

## Project Overview
This project focuses on developing a machine learning model to predict the onset of diabetes based on diagnostic measurements. The goal is to provide an accurate and reliable tool for early detection, which can significantly improve patient outcomes.

## Features
- **Data Preprocessing**: Handling missing values and outliers, feature scaling.
- **Machine Learning Models**: Utilizes various classification algorithms, with a focus on Random Forest for its robust performance.
- **Model Evaluation**: Comprehensive evaluation using metrics such as accuracy, precision, recall, and F1-score.
- **Deployment**: A simple Flask application for deploying the trained model, allowing for real-time predictions.

## Dataset
The project uses the Pima Indians Diabetes Database, which contains diagnostic measurements from female patients of Pima Indian heritage. The dataset includes features such as pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age, with an 'Outcome' variable indicating the presence or absence of diabetes.

## Technologies Used
- **Python**: Programming language.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical operations.
- **Matplotlib & Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning algorithms (e.g., RandomForestClassifier, StandardScaler, SimpleImputer, train_test_split, GridSearchCV).
- **Flask**: Web framework for deploying the machine learning model.
- **Joblib**: For saving and loading trained machine learning models and scalers.

## Installation and Setup
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: `venv\Scripts\activate`
   ```
3. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn Flask joblib
   ```

## Usage
### Jupyter Notebook Analysis
The `Diabetes_Project (1).ipynb` notebook contains the full data analysis, preprocessing, model training, and evaluation steps. You can run this notebook to understand the project in detail.

### Running the Flask Application
To run the web application for predictions:
1. Ensure you have the `app.py`, `diabetes_best_rf_model.pkl`, and `scaler.pkl` files in the same directory.
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. Open your web browser and navigate to `http://127.0.0.1:5000/` to access the prediction interface.

## Model Performance
The Random Forest Classifier achieved the following performance metrics on the test set:
- **Accuracy**: 98.5%
- **Precision**: 97.81%
- **Recall**: 97.81%
- **F1-Score**: 97.81%

## Contributing
Contributions are welcome! Please feel free to fork the repository, create a new branch, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details (if applicable, otherwise remove this section).


