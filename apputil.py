# apputil.py
import joblib
import pickle
import numpy as np

# ----------------------------
# Load Models
# ----------------------------
lr_model = joblib.load("model_1.pickle")      # price-only model
dtr_model = joblib.load("model_2.pickle")     # price+roast model

# ----------------------------
# Roast mapping helper
# ----------------------------
def roast_category(roast):
    mapping = {
        "Very Light": 0,
        "Light": 1,
        "Medium-Light": 2,
        "Medium": 3,
        "Medium-Dark": 4,
        "Dark": 5,
        "Very Dark": 6
    }
    return mapping.get(roast, None)

# ----------------------------
# Prediction Function
# ----------------------------
def predict_rating(df_X, text=False):
    """
    Predict coffee ratings.
    - When text=False: df_X must contain 100g_USD and roast.
    - When text=True: df_X must contain a column 'text'.
    """

    # ---- TEXT MODEL ----
    if text:
        # Load model_3 only when needed
        obj = pickle.load(open("model_3.pickle", "rb"))

        # Depending on how you saved model_3:
        if isinstance(obj, dict):
            vectorizer = obj["vectorizer"]
            model_text = obj["model"]
        else:
            vectorizer, model_text = obj  # tuple case

        X_vec = vectorizer.transform(df_X["text"].fillna(""))
        return model_text.predict(X_vec)

    # ---- NUMERIC + ROAST ----
    preds = []
    for _, row in df_X.iterrows():
        usd = row["100g_USD"]
        roast = row["roast"]
        roast_cat = roast_category(roast)

        if roast_cat is not None:
            preds.append(dtr_model.predict([[usd, roast_cat]])[0])
        else:
            preds.append(lr_model.predict([[usd]])[0])

    return preds
