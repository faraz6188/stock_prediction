import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the scaler and model
scaler = joblib.load('mms.h5')
model = load_model('stock_model.h5')

# Page configuration
st.set_page_config(page_title="📈 Stock Price Predictor", page_icon="💹", layout="wide")

# Header
def center_markdown(text):
    st.markdown(f"<h1 style='text-align: center; font-size: huge;'>{text}</h1>", unsafe_allow_html=True)

# Header
center_markdown("📊 Google Stock Price Prediction")

st.write("""
### Welcome to the **Google Stock Price Prediction App**! 🚀  
Enter stock price data using the sliders below 📉, and this app will predict future prices using an advanced LSTM model.  
### 💡 How It Works:
1️⃣ Use the sliders to input recent stock prices for the past **60 days**.  
2️⃣ Click **Predict** to calculate the stock price for Day 61.  
3️⃣ Visualize the predicted price below!  """)

# User input for stock prices
st.header("📉 Input Stock Prices")
st.write("Use the sliders below to input recent stock prices for the past **60 days**.")

# Create 10 columns for sliders
columns = st.columns(8)
prices = []

# Distribute sliders across the columns
for i in range(60):
    with columns[i % 8]:  # Assign sliders to columns in a round-robin manner
        price = st.slider(f"Day {i+1}", min_value=500.0, max_value=3000.0, value=1500.0, step=0.1, key=f"slider_{i+1}")
        prices.append(price)

# Predict stock prices
if st.button("💹 Predict Future Stock Prices"):
    # Convert input prices to a numpy array
    inputs = np.array(prices).reshape(-1, 1)
    inputs_scaled = scaler.transform(inputs)

    # Prepare the test data
    x_test = []
    x_test.append(inputs_scaled[:, 0])  # Use the user-provided 60 data points
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions_scaled = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions_scaled)

    # Display the results
    st.subheader("📊 Predicted Stock Price")
    st.success(f"📈 **Predicted Price for Day 61:** ${predictions.flatten()[0]:.2f}")
    st.balloons()

    note = {
            "⚠️ Note": "This model is not trained well enough to provide an exact stock value prediction. Due to the high computational cost, the training time for this model is around 1 to 2 hours, which is not feasible for me as a beginner with limited resources.Therefore, the predictions should be viewed as an approximation, and I recommend using more sophisticated models or a larger dataset for more accurate results.  Thank you for understanding! Stay tuned for future improvements!",
        }
    for question, answer in note.items():
            with st.expander(question):
                st.markdown(f"""
                <div style="background-color: red; padding: 10px; border-radius: 5px;">
                    <p style="color: white;">{answer}</p>
                </div>
                """, unsafe_allow_html=True)
# Footer
def footer_markdown(text):
    st.markdown(f"<h3 style='text-align: center; font-size: huge;'>{text}</h3>", unsafe_allow_html=True)
footer_markdown("""
© Copyright 2024 - Made by Mohammed Faraz Ali.  
""")