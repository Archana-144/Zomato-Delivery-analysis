import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- Page Title ---
st.title("🍴 Zomato Late Delivery Prediction Dashboard")
st.write("Predict whether a food order will be **late or on-time** based on key delivery factors.")

# --- Load or Create Data ---
data = {
    'Order_ID': range(1, 21),
    'Restaurant_Name': ['Burger Hub', 'Pizza Point', 'Café Delhi', 'Tandoor Treat', 'Burger Hub',
                        'Pizza Point', 'Rolls King', 'Subway', 'Tandoor Treat', 'Cake Walk',
                        'Domino’s', 'Taco Bell', 'Rolls King', 'Subway', 'Café Delhi',
                        'Domino’s', 'Pizza Point', 'Cake Walk', 'Burger Hub', 'Tandoor Treat'],
    'Distance_km': [2.5, 4.2, 3.1, 5.6, 6.3, 3.8, 2.9, 1.5, 4.8, 5.2, 3.0, 4.1, 2.3, 1.9, 3.7, 6.8, 4.4, 5.0, 2.2, 4.6],
    'Delivery_Time': [32, 48, 45, 60, 70, 42, 35, 28, 52, 58, 40, 46, 30, 25, 44, 68, 50, 55, 33, 53],
    'Weather': ['Clear', 'Rainy', 'Cloudy', 'Rainy', 'Rainy', 'Clear', 'Cloudy', 'Clear', 'Rainy', 'Rainy',
                'Clear', 'Cloudy', 'Clear', 'Clear', 'Cloudy', 'Rainy', 'Rainy', 'Cloudy', 'Clear', 'Rainy'],
    'Traffic': ['Low', 'High', 'Medium', 'High', 'High', 'Low', 'Medium', 'Low', 'High', 'Medium',
                'Low', 'High', 'Medium', 'Low', 'Medium', 'High', 'High', 'Medium', 'Low', 'High'],
    'Order_Type': ['Online', 'Online', 'Dine-In', 'Online', 'Online', 'Online', 'Dine-In', 'Online', 'Online', 'Online',
                   'Online', 'Online', 'Dine-In', 'Online', 'Dine-In', 'Online', 'Online', 'Dine-In', 'Online', 'Online'],
    'Restaurant_Rating': [4.5, 3.8, 4.2, 3.6, 3.5, 4.7, 4.0, 4.8, 3.9, 3.7, 4.3, 4.1, 4.5, 4.9, 4.0, 3.6, 3.8, 3.9, 4.6, 3.7]
}

df = pd.DataFrame(data)
df['Is_Late'] = df['Delivery_Time'] > 45

# --- Encode and Train Model ---
df_encoded = pd.get_dummies(df, columns=['Weather', 'Traffic', 'Order_Type'], drop_first=True)
X = df_encoded.drop(['Order_ID', 'Restaurant_Name', 'Is_Late', 'Delivery_Time'], axis=1)
y = df_encoded['Is_Late']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# --- Sidebar Inputs ---
st.sidebar.header("📦 Enter Delivery Details")

distance = st.sidebar.slider("Distance (in km)", 0.5, 10.0, 3.0)
rating = st.sidebar.slider("Restaurant Rating", 1.0, 5.0, 4.0)
weather = st.sidebar.selectbox("Weather Condition", ['Clear', 'Cloudy', 'Rainy'])
traffic = st.sidebar.selectbox("Traffic Condition", ['Low', 'Medium', 'High'])
order_type = st.sidebar.selectbox("Order Type", ['Online', 'Dine-In'])

# --- Convert Inputs into DataFrame ---
input_data = pd.DataFrame({
    'Distance_km': [distance],
    'Restaurant_Rating': [rating],
    'Weather_Cloudy': [1 if weather == 'Cloudy' else 0],
    'Weather_Rainy': [1 if weather == 'Rainy' else 0],
    'Traffic_Medium': [1 if traffic == 'Medium' else 0],
    'Traffic_High': [1 if traffic == 'High' else 0],
    'Order_Type_Online': [1 if order_type == 'Online' else 0]
})
# --- Align input columns with training columns ---
input_data = input_data.reindex(columns=X.columns, fill_value=0)


# --- Predict ---
prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0][1]

# --- Display Result ---
st.subheader("🚚 Prediction Result:")
if prediction:
    st.error(f"⚠️ The delivery is likely to be **Late** (Confidence: {prob*100:.2f}%)")
else:
    st.success(f"✅ The delivery is likely to be **On Time** (Confidence: {(1-prob)*100:.2f}%)")

# --- Footer ---
st.markdown("---")
st.caption("Developed by Archana B Y | Business Analytics Project 2025")
