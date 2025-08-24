import streamlit as st
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

st.title("House Price Prediction")

# Input fields
LotArea = st.number_input("Lot Area (sq ft)", min_value=1000.0, step=10.0)
OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
TotalBsmtSF = st.number_input("Basement Area (sq ft)", min_value=0.0, step=10.0)
GrLivArea = st.number_input("Living Area (sq ft)", min_value=300.0, step=10.0)
GarageCars = st.slider("Garage Capacity (cars)", 0, 5, 2)
GarageArea = st.number_input("Garage Area (sq ft)", min_value=0.0, step=10.0)

Neighborhood = st.selectbox("Neighborhood", ["CollgCr", "OldTown", "NridgHt", "NoRidge", "Gilbert"])
HouseStyle = st.selectbox("House Style", ["1Story", "2Story", "SFoyer", "SLvl", "1.5Fin", "2.5Fin"])
ExterQual = st.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa", "Po"])
KitchenQual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa", "Po"])

if st.button("Predict House Price"):
    data = CustomData(
        LotArea=LotArea,
        OverallQual=OverallQual,
        YearBuilt=YearBuilt,
        TotalBsmtSF=TotalBsmtSF,
        GrLivArea=GrLivArea,
        GarageCars=GarageCars,
        GarageArea=GarageArea,
        Neighborhood=Neighborhood,
        HouseStyle=HouseStyle,
        ExterQual=ExterQual,
        KitchenQual=KitchenQual,
    )

    df = data.get_data_as_dataframe()

    pipeline = PredictPipeline()
    prediction = pipeline.predict(df)[0]

    USD_TO_INR = 83.3
    price_in_inr = round(prediction * USD_TO_INR, 2)

    st.success(f"Estimated House Price: ₹{price_in_inr}")
