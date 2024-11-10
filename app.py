import streamlit as st
import numpy as np
import joblib
import pandas as pd



# Load your trained model
model = joblib.load("lr_model.pkl")  # Replace "model.pkl" with the path to your model file


# Streamlit app title
#st.title("Heart Attack Prediction")
st.title("Heart Attack Prediction")
st.caption("A Machine Learning Web App devloped by Indranil Bhattacharyya")

# User input features
st.sidebar.header("Input Parameters")

def user_input_features():
    Gender = st.sidebar.selectbox("Gender (Male=1, Female=0)", [1, 0])
    Age = st.sidebar.slider("Age", 0, 100, 50)
    HasTension = st.sidebar.selectbox("Has Tension (Yes=1, No=0)", [1, 0])
    AnyHeartDisease = st.sidebar.selectbox("Any Heart Disease (Yes=1, No=0)", [1, 0])
    NeverMarried = st.sidebar.selectbox("Never Married (Yes=1, No=0)", [1, 0])
    
    # Occupation Encodings
    Occupation_Govt_job = st.sidebar.selectbox("Government Job (Yes=1, No=0)", [1, 0])
    Occupation_Never_worked = st.sidebar.selectbox("Never Worked (Yes=1, No=0)", [1, 0])
    Occupation_Private = st.sidebar.selectbox("Private Job (Yes=1, No=0)", [1, 0])
    Occupation_Self_employed = st.sidebar.selectbox("Self-employed (Yes=1, No=0)", [1, 0])
    Occupation_children = st.sidebar.selectbox("Is a child (Yes=1, No=0)", [1, 0])
    
    LivesIn = st.sidebar.selectbox("Lives In (Rural=1, Urban=0)", [1, 0])
    GlucoseLevel = st.sidebar.slider("Glucose Level", 0.0, 1.0, 0.5)
    BMI = st.sidebar.slider("BMI", -1.0, 2.0, 0.0)
    
    # Smoking Status Encodings
    SmokingStatus_formerly_smoked = st.sidebar.selectbox("Formerly Smoked (Yes=1, No=0)", [1, 0])
    SmokingStatus_never_smoked = st.sidebar.selectbox("Never Smoked (Yes=1, No=0)", [1, 0])
    SmokingStatus_smokes = st.sidebar.selectbox("Currently Smokes (Yes=1, No=0)", [1, 0])

    data = {
        'Gender': Gender,
        'Age': Age,
        'HasTension': HasTension,
        'AnyHeartDisease': AnyHeartDisease,
        'NeverMarried': NeverMarried,
        'Occupation_Govt_job': Occupation_Govt_job,
        'Occupation_Never_worked': Occupation_Never_worked,
        'Occupation_Private': Occupation_Private,
        'Occupation_Self-employed': Occupation_Self_employed,
        'Occupation_children': Occupation_children,
        'LivesIn': LivesIn,
        'GlucoseLevel': GlucoseLevel,
        'BMI': BMI,
        'SmokingStatus_formerly_smoked': SmokingStatus_formerly_smoked,
        'SmokingStatus_never_smoked': SmokingStatus_never_smoked,
        'SmokingStatus_smokes': SmokingStatus_smokes,
    }

    features = np.array(list(data.values())).reshape(1, -1)
    return features, data

input_df, front_data = user_input_features()

# Display input parameters
st.subheader("User Input parameters")
front_df = pd.DataFrame([front_data])

st.write(front_df)

# Predict button
if st.button("Predict Heart Attack"):
    # Perform prediction
    prediction = model.predict(input_df)

    # Display prediction result
    if prediction[0] == 1:
        st.write("The model predicts a **high risk of heart attack**.")
    else:
        st.write("The model predicts a **low risk of heart attack**.")
    
    
