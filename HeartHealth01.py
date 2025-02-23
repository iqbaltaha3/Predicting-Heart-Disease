import streamlit as st
import pickle
import pandas as pd
import traceback

def load_model():
    """
    Loads the pre-trained Random Forest model from 'rf_model.pkl'.
    The pickle file should be a dictionary with keys:
    'model', 'scaler', and 'features'.
    """
    try:
        with open("rf_model.pkl", "rb") as file:
            model_data = pickle.load(file)
        if isinstance(model_data, dict):
            model = model_data.get("model", None)
            scaler = model_data.get("scaler", None)
            features = model_data.get("features", None)
        else:
            model = model_data
            scaler = None
            features = None

        if model is None:
            st.error("Model could not be loaded from the pickle file.")
            return None, None, None

        if features is None:
            if hasattr(model, "n_features_in_"):
                n_features = model.n_features_in_
                features = [f"feature_{i+1}" for i in range(n_features)]
                st.warning(f"Features not provided in pickle file. Using default names: {features}")
            else:
                st.error("Features are not available and cannot be inferred from the model.")
                return None, None, None

        st.success("Pre-trained model loaded successfully!")
        return model, scaler, features

    except Exception as e:
        st.error("Error loading pre-trained model: " + str(e))
        st.write(traceback.format_exc())
        return None, None, None

def main():
    st.title("Heart Disease Prediction")
    st.write("Enter the required feature values below to get a prediction using the pre-trained Random Forest model.")

    model, scaler, features = load_model()
    if model is None:
        st.error("Model could not be loaded. Please check your model file.")
        return

    # Define categorical mapping with original category names.
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

    # Collect user inputs.
    user_inputs = {}
    for feat in features:
        try:
            if feat in categorical_mapping:
                mapping = categorical_mapping[feat]
                options = list(mapping.values())
                selected = st.selectbox(f"Select value for {feat}:", options)
                # Convert the selected category back to its numeric code.
                encoded = [k for k, v in mapping.items() if v == selected][0]
                user_inputs[feat] = encoded
            else:
                user_inputs[feat] = st.number_input(f"Enter value for {feat}:", value=0.0, format="%.2f")
        except Exception as e:
            st.error(f"Error with input for {feat}: {e}")

    # Make a prediction when the button is pressed.
    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([user_inputs])
            if scaler is not None:
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df.values

            prediction = model.predict(input_scaled)[0]
            result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
            st.success(f"Prediction: {result}")
        except Exception as e:
            st.error("Error during prediction: " + str(e))
            st.write(traceback.format_exc())

if __name__ == "__main__":
    main()









