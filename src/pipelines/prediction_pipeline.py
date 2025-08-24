import os
import sys
import pandas as pd
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)
            return model.predict(data_scaled)

        except Exception as e:
            logging.error("Exception occurred in prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        LotArea: float,
        OverallQual: int,
        YearBuilt: int,
        TotalBsmtSF: float,
        GrLivArea: float,
        GarageCars: int,
        GarageArea: float,
        Neighborhood: str,
        HouseStyle: str,
        ExterQual: str,
        KitchenQual: str,
    ):
        self.LotArea = LotArea
        self.OverallQual = OverallQual
        self.YearBuilt = YearBuilt
        self.TotalBsmtSF = TotalBsmtSF
        self.GrLivArea = GrLivArea
        self.GarageCars = GarageCars
        self.GarageArea = GarageArea
        self.Neighborhood = Neighborhood
        self.HouseStyle = HouseStyle
        self.ExterQual = ExterQual
        self.KitchenQual = KitchenQual

    def get_data_as_dataframe(self):
        try:
            data = {
                "LotArea": [self.LotArea],
                "OverallQual": [self.OverallQual],
                "YearBuilt": [self.YearBuilt],
                "TotalBsmtSF": [self.TotalBsmtSF],
                "GrLivArea": [self.GrLivArea],
                "GarageCars": [self.GarageCars],
                "GarageArea": [self.GarageArea],
                "Neighborhood": [self.Neighborhood],
                "HouseStyle": [self.HouseStyle],
                "ExterQual": [self.ExterQual],
                "KitchenQual": [self.KitchenQual],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)
