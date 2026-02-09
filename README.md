# Student Exam Performance Prediction

This project is an end-to-end Machine Learning web application that predicts student exam performance based on various demographic and educational factors.

## Project Overview

The goal is to understand how a student's performance (specifically in Mathematics) is affected by variables such as Gender, Ethnicity, Parental Level of Education, Lunch, and Test Preparation Course. The project implements a complete machine learning pipeline from data ingestion to deployment as a web application.

## Project Flow & Pipeline

The project follows a modular architecture:

1.  **Data Ingestion** (`src/components/data_ingestion.py`):
    - Reads the raw data from the source.
    - Splits the data into training and testing sets.
    - Saves the raw, train, and test data paths.

2.  **Data Transformation** (`src/components/data_transformation.py`):
    - Handles missing values.
    - Performs one-hot encoding for categorical variables.
    - Standardizes numerical features using `StandardScaler`.
    - Saves the preprocessor object as a pickle file.

3.  **Model Training** (`src/components/model_trainer.py`):
    - Trains multiple machine learning models.
    - Evaluates them based on R2 score.
    - Selects the best performing model (e.g., Linear Regression, Decision Tree, etc.).
    - Saves the best model as a pickle file.

4.  **Prediction Pipeline** (`src/pipeline/predict_pipeline.py`):
    - `CustomData` class: Maps HTML form input to a DataFrame.
    - `PredictPipeline` class: Loads the saved model and preprocessor to predict the target variable.

5.  **Web Application** (`app.py`):
    - A Flask web server that handles user requests.
    - Renders the UI (`templates/`).
    - Takes user input and calls the prediction pipeline.

## Folder Structure

```
mlproject/
├── .ebextensions/       # AWS Elastic Beanstalk configuration
├── artifacts/           # Stores models and data (train.csv, test.csv, model.pkl, etc.)
├── catboost_info/       # CatBoost model logs
├── logs/                # Application logs
├── notebook/            # Jupyter notebooks for EDA and experimentation
├── src/                 # Source code
│   ├── components/      # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/        # Training and prediction pipelines
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── utils/           # Utility functions (save/load object, etc.)
│   ├── exception.py     # Custom exception handling
│   └── logger.py        # Logging configuration
├── static/              # Static files (CSS, JS, Images)
├── templates/           # HTML templates
│   ├── home.html        # Prediction page
│   └── index.html       # Landing page
├── app.py               # Flask application entry point
├── application.py       # WSGI entry point
├── requirements.txt     # Python dependencies
├── setup.py             # Package installation script
└── README.md            # Project documentation
```

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Application**:
    ```bash
    python app.py
    ```

3.  **Access the App**:
    Open your browser and navigate to `http://127.0.0.1:5000`