from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy import sparse
import numpy as np
import pandas as pd
import os, sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_cols = [
                "LotArea",
                "OverallQual",
                "YearBuilt",
                "TotalBsmtSF",
                "GrLivArea",
                "GarageCars",
                "GarageArea",
            ]
            categorical_cols = [
                "Neighborhood",
                "HouseStyle",
                "ExterQual",
                "KitchenQual",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            return ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_cols),
                    ("cat", cat_pipeline, categorical_cols),
                ]
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target = "SalePrice"
            id_col = "Id"
            drop_cols = [target, id_col]

            X_train = train_df.drop(columns=drop_cols, axis=1)
            y_train = train_df[target]
            X_test = test_df.drop(columns=drop_cols, axis=1)
            y_test = test_df[target]

            preprocessor = self.get_data_transformation_object()

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            if sparse.issparse(X_train_arr):
                X_train_arr = X_train_arr.toarray()
            if sparse.issparse(X_test_arr):
                X_test_arr = X_test_arr.toarray()

            X_train_arr = np.asarray(X_train_arr)
            X_test_arr = np.asarray(X_test_arr)

            y_train_arr = y_train.values.reshape(-1, 1)
            y_test_arr = y_test.values.reshape(-1, 1)

            train_arr = np.hstack((X_train_arr, y_train_arr))
            test_arr = np.hstack((X_test_arr, y_test_arr))

            save_object(self.config.preprocessor_obj_file_path, preprocessor)

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys) from e
