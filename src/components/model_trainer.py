import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array,test_array):
        try:
            logging.info("Splitting Training and Testing Input Data")
            X_train,y_train,X_test,y_test= (
                train_array[:,:-1], # Take out Last element and store everything into X_train
                train_array[:,-1], #Take Last element as y_train
                test_array[:,:-1], # Same for X_train
                test_array[:,-1] # X_test
            )

            # below Models we would be trying
            models ={
                "Random forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Descent": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbour Regressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Classifier":AdaBoostRegressor()
            }

            model_report = evaluate_model(X_train=X_train,y_train = y_train,X_test = X_test,
                                          y_test=y_test, models=models)
            
            ## To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No Best Model found")
            logging.info(f"Best found model on both training and testing dataset")

            # Thie method will dump the best model into model.pkl file
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # predicted output for test data
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)