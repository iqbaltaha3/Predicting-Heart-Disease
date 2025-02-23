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

    # Define categorical mapping with original category labels.
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

    # Detailed explanations for each feature.
    feature_explanations = {
        "age": (
            "**What:** Age of the patient (in years).\n\n"
            "**Why:** Older age is associated with higher risk of heart disease.\n\n"
            "**When:** Recorded at the time of clinical evaluation.\n\n"
            "**Where:** Typically obtained from patient records.\n\n"
            "**How:** Self-reported or verified during a clinical visit."
        ),
        "sex": (
            "**What:** Biological gender of the patient.\n\n"
            "**Why:** Risk factors and prevalence of heart disease differ by sex.\n\n"
            "**When:** Recorded during patient intake.\n\n"
            "**Where:** Patient records or clinical forms.\n\n"
            "**How:** Coded as 0 (Female) or 1 (Male)."
        ),
        "cp": (
            "**What:** Type of chest pain experienced.\n\n"
            "**Why:** Different chest pain types indicate different levels of cardiac risk.\n\n"
            "**When:** Assessed during patient evaluation.\n\n"
            "**Where:** Clinical diagnosis.\n\n"
            "**How:** Encoded as 0 (Typical Angina), 1 (Atypical Angina), 2 (Non-Anginal Pain), or 3 (Asymptomatic)."
        ),
        "trestbps": (
            "**What:** Resting blood pressure (mm Hg).\n\n"
            "**Why:** High blood pressure is a known risk factor for heart disease.\n\n"
            "**When:** Measured at rest during clinical evaluation.\n\n"
            "**Where:** In a clinical setting using a sphygmomanometer.\n\n"
            "**How:** Recorded as a numeric value."
        ),
        "chol": (
            "**What:** Serum cholesterol level (mg/dl).\n\n"
            "**Why:** Elevated cholesterol is linked to increased heart disease risk.\n\n"
            "**When:** Measured after fasting.\n\n"
            "**Where:** In a laboratory test.\n\n"
            "**How:** Reported as a numeric value."
        ),
        "fbs": (
            "**What:** Fasting blood sugar > 120 mg/dl.\n\n"
            "**Why:** Indicates possible diabetes, which is a risk factor for heart disease.\n\n"
            "**When:** Measured after fasting.\n\n"
            "**Where:** In a clinical lab test.\n\n"
            "**How:** Coded as 0 (False) or 1 (True)."
        ),
        "restecg": (
            "**What:** Resting electrocardiographic results.\n\n"
            "**Why:** Provides information on heart's electrical activity and abnormalities.\n\n"
            "**When:** Taken at rest during evaluation.\n\n"
            "**Where:** In a clinical setting using an ECG machine.\n\n"
            "**How:** Encoded as 0 (Normal), 1 (ST-T Wave Abnormality), or 2 (Left Ventricular Hypertrophy)."
        ),
        "exang": (
            "**What:** Exercise-induced angina.\n\n"
            "**Why:** Indicates the heart's response to physical stress.\n\n"
            "**When:** Assessed during a stress test.\n\n"
            "**Where:** In a clinical setting.\n\n"
            "**How:** Coded as 0 (No) or 1 (Yes)."
        ),
        "slope": (
            "**What:** Slope of the peak exercise ST segment.\n\n"
            "**Why:** Reflects the heart's response to exercise stress.\n\n"
            "**When:** Measured during an exercise test.\n\n"
            "**Where:** In a clinical ECG.\n\n"
            "**How:** Encoded as 0 (Upsloping), 1 (Flat), or 2 (Downsloping)."
        ),
        "ca": (
            "**What:** Number of major vessels (0-3) colored by fluoroscopy.\n\n"
            "**Why:** Indicates the extent of arterial blockage in the heart.\n\n"
            "**When:** Determined during an angiographic exam.\n\n"
            "**Where:** In a hospital setting during diagnostic imaging.\n\n"
            "**How:** Reported as a numeric code from 0 to 3."
        ),
        "thal": (
            "**What:** Thalassemia status.\n\n"
            "**Why:** Abnormalities in blood can affect oxygen transport, influencing heart risk.\n\n"
            "**When:** Assessed during blood tests.\n\n"
            "**Where:** In a clinical laboratory.\n\n"
            "**How:** Encoded as 0 (Normal), 1 (Fixed Defect), 2 (Reversible Defect), or 3 (Unknown)."
        )
    }

    # Collect user inputs.
    user_inputs = {}
    for feat in features:
        try:
            if feat in categorical_mapping:
                mapping = categorical_mapping[feat]
                options = list(mapping.values())
                selected = st.selectbox(f"Select value for **{feat}**:", options, key=feat)
                # Convert selected label back to its encoded value.
                encoded = [k for k, v in mapping.items() if v == selected][0]
                user_inputs[feat] = encoded
            else:
                user_inputs[feat] = st.number_input(f"Enter value for **{feat}**:", value=0.0, format="%.2f", key=feat)
            # Add an expander for detailed explanation.
            with st.expander(f"Know more about **{feat}**"):
                explanation = feature_explanations.get(feat, "No detailed explanation available for this feature.")
                st.markdown(explanation)
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










