from flask import Flask, render_template, request
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("form.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html")

    data = CustomData(
        LotArea      = float(request.form["LotArea"]),
        OverallQual  = int(request.form["OverallQual"]),
        YearBuilt    = int(request.form["YearBuilt"]),
        TotalBsmtSF  = float(request.form["TotalBsmtSF"]),
        GrLivArea    = float(request.form["GrLivArea"]),
        GarageCars   = int(request.form["GarageCars"]),
        GarageArea   = float(request.form["GarageArea"]),
        Neighborhood = request.form["Neighborhood"],
        HouseStyle   = request.form["HouseStyle"],
        ExterQual    = request.form["ExterQual"],
        KitchenQual  = request.form["KitchenQual"],
)


    df = data.get_data_as_dataframe()
    pipeline = PredictPipeline()
    prediction = pipeline.predict(df)[0]
    price = round(prediction, 2)

    USD_TO_INR = 83.3
    price_in_inr = round(prediction * USD_TO_INR, 2)


    return render_template("result.html", final_result=price_in_inr, currency='₹')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
