import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
filename = 'calories_burned_model.pkl'
try:
    loaded_model = pickle.load(open(filename, 'rb'))
except FileNotFoundError:
    st.error(f"Error: The model file '{filename}' was not found. Make sure it's in the same directory as the app.")
    st.stop()

# Initialize LabelEncoder (make sure it's the same as used during training)
le = LabelEncoder()
le.fit(['female', 'male']) # Fit with the original categories

# Set the title of the app
st.title('Calories Burned Predictor')

st.sidebar.header("User Input Parameters: ")

def user_input_features():
    duration = st.sidebar.number_input('Duration (minutes)', min_value=1, max_value=60, value=30)
    heart_rate = st.sidebar.number_input('Heart Rate (bpm)', min_value=60, max_value=180, value=100)
    body_temp = st.sidebar.number_input('Body Temperature (°C)', min_value=36.0, max_value=42.0, value=39.0)
    weight = st.sidebar.number_input('Weight (kg)', min_value=30, max_value=150, value=70)
    age = st.sidebar.number_input('Age (years)', min_value=18, max_value=100, value=30)
    gender_str = st.sidebar.selectbox('Gender', ['female', 'male'])

    # Convert gender to numerical encoding
    gender = le.transform([gender_str])[0]

    # Create a DataFrame from the input features
    input_data = pd.DataFrame({
        'Duration': [duration],
        'Heart_Rate': [heart_rate],
        'Body_Temp': [body_temp],
        'Weight': [weight],
        'Age': [age],
        'Gender': [gender]
    })
    return input_data

user_data = user_input_features()

st.write("---")
st.header("Your Parameters: ")
st.write(user_data)

# Create a button to trigger the prediction
if st.button('Predict Calories Burned'):
    # Make the prediction
    prediction = loaded_model.predict(user_data)[0]

    # Display the prediction
    st.success(f'Predicted Calories Burned: {prediction:.2f}')

st.write("---")
st.write("This app uses a pre-trained model to predict calories burned based on your input parameters.")