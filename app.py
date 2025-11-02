import streamlit as st
import pandas as pd
from apputil import predict_rating

st.set_page_config(page_title="Coffee Rating Predictor")

# ------------------------------------------------------
#                   APP UI HEADER
# ------------------------------------------------------

st.title("☕ Coffee Rating Predictor")
st.write("Predict coffee ratings using price, roast type, or text description.")

# ------------------------------------------------------
#                SELECT PREDICTION MODE
# ------------------------------------------------------

choice = st.radio(
    "Choose input type:",
    ["Price Only", "Price + Roast", "Text Review"]
)

# ------------------------------------------------------
#          OPTION 1 — PRICE ONLY (model_1)
# ------------------------------------------------------

if choice == "Price Only":
    st.subheader("Predict Rating Using Price Only")

    price = st.number_input("Enter price (100g_USD):", min_value=0.0, step=0.5)

    if st.button("Predict Rating"):
        df = pd.DataFrame([[price, None]], columns=["100g_USD", "roast"])
        prediction = predict_rating(df)
        st.success(f"Predicted Rating: **{prediction[0]:.2f}**")

# ------------------------------------------------------
#         OPTION 2 — PRICE + ROAST (model_2)
# ------------------------------------------------------

elif choice == "Price + Roast":
    st.subheader("Predict Rating Using Price + Roast")

    price = st.number_input("Enter price (100g_USD):", min_value=0.0, step=0.5)

    roast_list = [
        "Very Light", "Light", "Medium-Light", 
        "Medium", "Medium-Dark", "Dark", "Very Dark"
    ]

    roast = st.selectbox("Select roast type:", roast_list)

    if st.button("Predict Rating"):
        df = pd.DataFrame([[price, roast]], columns=["100g_USD", "roast"])
        prediction = predict_rating(df)
        st.success(f"Predicted Rating: **{prediction[0]:.2f}**")

# ------------------------------------------------------
#         OPTION 3 — TEXT REVIEW (model_3)
# ------------------------------------------------------

elif choice == "Text Review":
    st.subheader("Predict Rating Using Review Text")

    text_input = st.text_area(
        "Enter a coffee review:",
        "A delightful coffee with a smooth taste."
    )

    if st.button("Predict from Text"):
        df = pd.DataFrame([text_input], columns=["text"])
        prediction = predict_rating(df, text=True)
        st.success(f"Predicted Rating: **{prediction[0]:.2f}**")
