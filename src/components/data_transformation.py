import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils.common import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """This function is responsible for data transformation"""
        try:
            logging.info("Creating data transformer object")

            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education',
                'lunch', 
                'test_preparation_course']    
              
            # Numerical pipeline
            num_pipeline = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])   
            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ]) 
            logging.info("Numerical and categorical pipelines created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )     
            return preprocessor
        
        except Exception as e:
            logging.error("Error occurred while creating data transformer object")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Training and testing data loaded")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            # Separate input features and target feature for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target feature for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply transformation on training and testing data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Data transformation completed")

            # Combine transformed input features with target feature
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException(e, sys)