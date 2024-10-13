import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load the preprocessor and models
preprocessor = joblib.load('preprocessor_pipeline.pkl')
dnn_model = load_model('dnn_model.h5')
ada_model = joblib.load('adaboost_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')

# Prediction function
def predict_parking(model, input_data):
    # Make predictions using the selected model
    prediction = model.predict(input_data)
    return prediction

# Output interpretation function
def interpret_dnn_output(predictions):
    result = []
    for i, pred in enumerate(predictions):
        if pred >= 0.5:
            result.append(("Business has parking and parking is validated.", "success"))
        else:
            result.append(("Business does not have parking or parking is not validated.", "error"))
    return result

# Streamlit app
st.title("🚗 Business Parking Prediction System")
st.markdown("""
### Welcome to the Business Parking Prediction App! 
Upload a CSV file with your business data to find out if the business has parking and whether it's validated. You can choose between **DNN**, **Adaboost**, or **XGBoost** models.
""")

# File uploader
st.write("#### Upload your input file in CSV format:")
uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    input_data = pd.read_csv(uploaded_file)
    
    st.write("### 📋 Preview of the uploaded data:")
    st.write(input_data.head())
    
    # Preprocess the data using the loaded preprocessor
    try:
        input_data_encoded = preprocessor.transform(input_data)
    except Exception as e:
        st.error(f"❌ Error in preprocessing the input data: {e}")
        st.stop()

    # Ensure the input shape matches the model's expected input shape
    if input_data_encoded.shape[1] != dnn_model.input_shape[1]:
        st.error(f"❌ Input shape mismatch. Expected {dnn_model.input_shape[1]} features, but got {input_data_encoded.shape[1]}.")
        st.stop()

    # Model selection dropdown
    model_choice = st.selectbox("Select the Model", ["DNN", "Adaboost", "XGBoost"])

    if st.button("🔍 Predict"):
        # Perform prediction
        if model_choice == "DNN":
            predictions = predict_parking(dnn_model, input_data_encoded)
        elif model_choice == "Adaboost":
            predictions = predict_parking(ada_model, input_data_encoded)
        else:
            predictions = predict_parking(xgb_model, input_data_encoded)

        # Convert predictions to binary if necessary (for DNN and XGBoost)
        if model_choice in ["DNN", "XGBoost"]:
            predictions = (predictions > 0.5).astype(int)

        # Beautify predictions
        st.write("### 🔍 Predictions and Interpretations:")
        interpretation = interpret_dnn_output(predictions)
        
        # Show predictions with color-coded success or error messages
        for i, (text, status) in enumerate(interpretation):
            if status == "success":
                st.success(f"Sample {i + 1}: {text}")
            else:
                st.error(f"Sample {i + 1}: {text}")
        
        # Create a DataFrame for download
        result_df = pd.DataFrame({
            "Prediction": predictions.flatten(),
            "Interpretation": [text for text, _ in interpretation]
        })

        # Display the table
        st.write("### 📊 Results Table")
        st.table(result_df)

        # Provide a download button for the results
        csv = result_df.to_csv(index=False)
        st.download_button(label="📥 Download Prediction Results", data=csv, file_name="predictions.csv", mime="text/csv")
