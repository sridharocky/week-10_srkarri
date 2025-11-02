# train_text.py
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

# ----------------------------
# Load data from GitHub URL
# ----------------------------
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

print(f"✅ Data loaded successfully! {df.shape[0]} rows, {df.shape[1]} columns")

# ----------------------------
# Prepare text and target
# ----------------------------
X_text = df["desc_3"].fillna("")
y = df["rating"]

# ----------------------------
# TF-IDF vectorization
# ----------------------------
vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
X_vec = vectorizer.fit_transform(X_text)

lr_text = LinearRegression()
lr_text.fit(X_vec, y)

# ----------------------------
# Save both model and vectorizer
# ----------------------------
with open("model_3.pickle", "wb") as f:
    pickle.dump((vectorizer, lr_text), f)

print("✅ model_3.pickle saved successfully!")

# train_model_2.py
import pandas as pd
import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Load dataset
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# Roast mapping function
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
    return mapping.get(roast, np.nan)

# Create roast_cat
df["roast_cat"] = df["roast"].apply(roast_category)

# Drop rows missing required fields
df = df.dropna(subset=["100g_USD", "rating"])

# Features & target
X = df[["100g_USD", "roast_cat"]]
y = df["rating"]

# Train
model = DecisionTreeRegressor()
model.fit(X, y)

# Save
joblib.dump(model, "model_2.pickle")
print("✅ model_2.pickle saved successfully!")

