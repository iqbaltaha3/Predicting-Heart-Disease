import streamlit as st
import pickle
import pandas as pd
import traceback

def load_model():
    """
    Loads the pre-trained Random Forest model from 'rf_model.pkl'.
    The pickle file is expected to be a dictionary with keys:
    'model', 'scaler', and 'features'.
    """
    try:
        with open("rf_model.pkl", "rb") as file:
            model_data = pickle.load(file)
        
        # Check if the loaded object is a dictionary.
        if isinstance(model_data, dict):
            model = model_data.get("model", None)
            scaler = model_data.get("scaler", None)
            features = model_data.get("features", None)
        else:
            # Fallback: if it's not a dictionary, assume it's just the model.
            model = model_data
            scaler = None
            features = None
        
        if model is None:
            st.error("Model could not be loaded from the pickle file.")
            return None, None, None
        
        # If features are not provided, attempt to infer feature names.
        if features is None:
            if hasattr(model, "n_features_in_"):
                n_features = model.n_features_in_
                features = [f"feature_{i+1}" for i in range(n_features)]
                st.warning(f"Features not provided in pickle file. Using default feature names: {features}")
            else:
                st.error("Features are not available and cannot be inferred from the model.")
                return None, None, None
        
        st.success("Pre-trained model loaded successfully!")
        return model, scaler, features

    except Exception as e:
        st.error("Error loading pre-trained model: " + str(e))
        st.write(traceback.format_exc())
        return None, None, None

def prediction_section():
    st.title("Heart Disease Prediction")
    st.write("Enter the required feature values to get a prediction using the pre-trained Random Forest model.")

    # Load the model, scaler, and feature list.
    model, scaler, features = load_model()
    if model is None:
        st.error("Model could not be loaded. Please check the model file and try again.")
        return

    # Define mapping for categorical columns with original categories.
    # Update these mappings as per your dataset if needed.
    categorical_mapping = {
        "sex": {0: "Female", 1: "Male"},
        "cp": {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"},
        "fbs": {0: "False", 1: "True"},
        "restecg": {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"},
        "exang": {0: "No", 1: "Yes"},
        "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
        "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
        "thal": {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Unknown"}
    }

    # Collect user inputs for each feature.
    user_inputs = {}
    for feat in features:
        try:
            if feat in categorical_mapping:
                # Create a dropdown with the original category labels.
                mapping = categorical_mapping[feat]
                options = list(mapping.values())
                selected_label = st.selectbox(f"Select value for {feat}:", options=options)
                # Convert the selected label back to its encoded value.
                encoded_value = [k for k, v in mapping.items() if v == selected_label][0]
                user_inputs[feat] = encoded_value
            else:
                # For numerical features, use a number input.
                user_input = st.number_input(f"Enter value for {feat}:", value=0.0, format="%.2f")
                user_inputs[feat] = user_input
        except Exception as e:
            st.error(f"Error with input for {feat}: {e}")

    # Make a prediction when the button is pressed.
    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([user_inputs])
            # If a scaler is provided, transform the input.
            if scaler is not None:
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df.values  # Assume no scaling is needed.
                
            prediction = model.predict(input_scaled)
            result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
            st.success(f"Prediction: {result}")
        except Exception as e:
            st.error("Error during prediction: " + str(e))
            st.write(traceback.format_exc())

def main():
    prediction_section()

if __name__ == "__main__":
    main()








