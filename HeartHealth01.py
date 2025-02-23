import streamlit as st
import pickle
import pandas as pd
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

# Custom CSS for improved aesthetics.
st.markdown(
    """
    <style>
    /* Overall background */
    .reportview-container {
        background: #f0f2f6;
    }
    /* Title styling */
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #333333;
    }
    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 5px;
    }
    /* Sidebar title styling */
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def load_model():
    """
    Loads the pre-trained Random Forest model from 'rf_model.pkl'.
    The pickle file is expected to be a dictionary with keys:
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
    st.header("Heart Disease Prediction")
    st.write("Enter the required feature values to get a prediction using the pre-trained Random Forest model.")
    
    model, scaler, features = load_model()
    if model is None:
        st.error("Model could not be loaded. Please check the model file and try again.")
        return

    # Define a mapping for categorical columns with original category labels.
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
    
    user_inputs = {}
    for feat in features:
        try:
            if feat in categorical_mapping:
                mapping = categorical_mapping[feat]
                options = list(mapping.values())
                selected_label = st.selectbox(f"Select value for {feat}:", options=options)
                encoded_value = [k for k, v in mapping.items() if v == selected_label][0]
                user_inputs[feat] = encoded_value
            else:
                user_input = st.number_input(f"Enter value for {feat}:", value=0.0, format="%.2f")
                user_inputs[feat] = user_input
        except Exception as e:
            st.error(f"Error with input for {feat}: {e}")
    
    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([user_inputs])
            if scaler is not None:
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df.values
            
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_scaled)[0]
                prediction = model.predict(input_scaled)[0]
                result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
                st.success(f"Prediction: {result}")
                st.write("Prediction probabilities:", probabilities)
            else:
                prediction = model.predict(input_scaled)[0]
                result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
                st.success(f"Prediction: {result}")
        except Exception as e:
            st.error("Error during prediction: " + str(e))
            st.write(traceback.format_exc())

def visualization_section():
    st.header("Data Visualizations")
    st.write("Upload your heart disease dataset (CSV) to view simple, relevant visualizations.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Dataset Overview")
            st.dataframe(df.head())
            
            st.subheader("Distribution of Heart Disease")
            fig1, ax1 = plt.subplots()
            df["target"].value_counts().plot(kind="bar", color=["#4CAF50", "#F44336"], ax=ax1)
            ax1.set_title("Heart Disease Distribution (0 = No, 1 = Yes)")
            ax1.set_xlabel("Target")
            ax1.set_ylabel("Count")
            st.pyplot(fig1)
            
            st.subheader("Age Distribution")
            fig2, ax2 = plt.subplots()
            ax2.hist(df["age"], bins=20, color="#2196F3", edgecolor="black")
            ax2.set_title("Age Distribution")
            ax2.set_xlabel("Age")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)
            
            st.subheader("Chest Pain Type Distribution")
            if "cp" in df.columns:
                fig3, ax3 = plt.subplots()
                df["cp"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax3)
                ax3.set_title("Chest Pain Type Distribution")
                st.pyplot(fig3)
        except Exception as e:
            st.error("Error processing the uploaded file: " + str(e))
            st.write(traceback.format_exc())
    else:
        st.info("Please upload a CSV file to see visualizations.")

def home_section():
    st.title("Heart Disease Analysis & Prediction App")
    st.write("""
        Welcome to the Heart Disease Analysis & Prediction App.  
        This application uses a pre-trained Random Forest model to predict heart disease based on several input features.  
        Use the sidebar to navigate between making predictions and viewing data visualizations.
    """)
    # Update the image URL below with an appropriate image URL or local image path.
    st.image("https://www.publicdomainpictures.net/pictures/30000/velka/heart-1451530032i2R.jpg", caption="Heart Health", use_column_width=True)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Prediction", "Visualizations"])
    
    if page == "Home":
        home_section()
    elif page == "Prediction":
        prediction_section()
    elif page == "Visualizations":
        visualization_section()
    else:
        st.error("Page not found.")

if __name__ == "__main__":
    main()









