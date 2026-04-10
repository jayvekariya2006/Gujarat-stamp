from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("gujarat_stamp_duty_cleaned.csv")

# =========================
# TARGET & FEATURES
# =========================
# ⚠️ check column name (important)
target_column = "stamp_dut"   # agar alag ho to change karna

y = df[target_column]
X = df.drop(target_column, axis=1)

# =========================
# HANDLE CATEGORICAL
# =========================
X = pd.get_dummies(X)

# =========================
# TRAIN MODEL
# =========================
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Save columns for prediction
columns = X.columns

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # FORM DATA
    data = dict(request.form)

    # Convert numeric values
    for key in data:
        try:
            data[key] = float(data[key])
        except:
            pass

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Match columns with training data
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)[0]

    return render_template("index.html", result=round(prediction, 2))


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)