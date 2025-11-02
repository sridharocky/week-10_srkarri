# ---------- train.py ----------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# ---------- STEP 1: LOAD DATA ----------
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
print("Loading data from:", url)
df = pd.read_csv(url)

print("✅ Data loaded successfully. Shape:", df.shape)
print(df.head())

# ---------- STEP 2: FEATURE SELECTION ----------
# For Exercise 1: rating ~ 100g_USD
df = df.dropna(subset=["100g_USD", "rating"])

X = df[["100g_USD"]]
y = df["rating"]

# ---------- STEP 3: SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- STEP 4: TRAIN ----------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------- STEP 5: EVALUATE ----------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"✅ Model trained successfully. MSE: {mse:.2f}")

# ---------- STEP 6: SAVE ----------
joblib.dump(model, "model_1.pickle")
print("💾 model_1.pickle saved successfully!")
