import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import traceback

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from xgboost import XGBClassifier

# Set Streamlit configuration for wide mode
st.set_page_config(layout="wide")

def load_data(file_path=None, uploaded_file=None):
    """Load the dataset either from an uploaded file or from a default file path."""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully!")
        elif file_path is not None:
            df = pd.read_csv(file_path)
            st.sidebar.info("Using default heart_disease.csv file.")
        else:
            st.error("No file provided.")
            return None
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        return df
    except Exception as e:
        st.error("Error loading data: " + str(e))
        st.write(traceback.format_exc())
        return None

def display_data_overview(df):
    st.header("Data Overview")
    try:
        st.subheader("First 5 Rows")
        st.dataframe(df.head())
    except Exception as e:
        st.error("Error displaying data head: " + str(e))
    
    try:
        st.subheader("Data Shape")
        st.write(df.shape)
    except Exception as e:
        st.error("Error displaying shape: " + str(e))
    
    try:
        st.subheader("Missing Values")
        st.write(df.isnull().sum())
    except Exception as e:
        st.error("Error computing missing values: " + str(e))
    
    try:
        st.subheader("Data Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    except Exception as e:
        st.error("Error displaying data info: " + str(e))
    
    try:
        st.subheader("Statistical Summary")
        st.write(df.describe())
    except Exception as e:
        st.error("Error displaying summary: " + str(e))

def perform_eda(df):
    st.header("Exploratory Data Analysis")
    # Duplicate removal info
    try:
        st.subheader("Duplicate Removal")
        duplicates_before = df.duplicated().sum()
        st.write(f"Duplicates before removal: {duplicates_before}")
        # Already removed in load_data; showing after removal
        duplicates_after = df.duplicated().sum()
        st.write(f"Duplicates after removal: {duplicates_after}")
    except Exception as e:
        st.error("Error during duplicate check: " + str(e))
    
    # Identify categorical and numerical columns
    try:
        cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        # Exclude 'target' from numeric columns
        num_cols = [col for col in df.columns if col not in cat_cols + ['target']]
    except Exception as e:
        st.error("Error defining column groups: " + str(e))
        return

    # Numeric column plots
    for col in num_cols:
        try:
            st.subheader(f"Histogram of {col}")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error plotting histogram for {col}: " + str(e))
        
        try:
            st.subheader(f"Violin Plot of {col} by Sex")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.violinplot(x='sex', y=col, data=df, color='green', ax=ax)
            ax.set_title(f"Violin Plot of {col} by Sex")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error plotting violin plot for {col}: " + str(e))
    
    # Categorical column plots
    for col in cat_cols:
        try:
            st.subheader(f"Countplot of {col}")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.countplot(x=col, data=df, ax=ax)
            ax.set_title(f"Countplot of {col}")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error plotting countplot for {col}: " + str(e))
    
    # Correlation heatmap
    try:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error("Error generating correlation heatmap: " + str(e))
    
    # Target countplot
    try:
        st.subheader("Countplot of Target (Disease Status)")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x='target', data=df, ax=ax)
        ax.set_title("Patients with and without Heart Disease")
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error("Error generating target countplot: " + str(e))
    
    # Age vs Disease Status Plots
    try:
        st.subheader("Age vs Disease Status - Stripplot")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.stripplot(x='target', y='age', data=df, palette='pastel', jitter=True, ax=ax)
        ax.set_title("Age vs Disease Status")
        ax.set_xticklabels(['No Disease', 'Disease'])
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error("Error generating stripplot: " + str(e))
    
    try:
        st.subheader("Age Distribution by Disease Status - Boxplot")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.boxplot(x='target', y='age', data=df, palette='pastel', ax=ax)
        ax.set_title("Age Distribution by Disease Status")
        ax.set_xticklabels(['No Disease', 'Disease'])
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error("Error generating boxplot: " + str(e))

def train_and_compare_models(df):
    st.header("Model Training & Comparison")
    try:
        st.subheader("Preprocessing")
        X = df.drop(columns=['target'])
        y = df['target']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        st.success("Preprocessing completed.")
    except Exception as e:
        st.error("Error during preprocessing: " + str(e))
        st.write(traceback.format_exc())
        return

    # Dictionary to hold metrics and predictions
    metrics_dict = {}
    predictions = {}

    # Logistic Regression
    try:
        st.subheader("Logistic Regression")
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        y_pred_lr = lr.predict(x_test)
        conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        report_lr = classification_report(y_test, y_pred_lr, output_dict=True)
        metrics_dict['Logistic Regression'] = {
            'Accuracy': accuracy_lr,
            'F1 Score': report_lr['1']['f1-score'],
            'Precision': report_lr['1']['precision'],
            'Recall': report_lr['1']['recall']
        }
        predictions['Logistic Regression'] = y_pred_lr
        st.write("Confusion Matrix:")
        st.write(conf_matrix_lr)
        st.write(f"Accuracy: {accuracy_lr:.2f}")
        st.write(f"F1 Score: {report_lr['1']['f1-score']:.2f}")
        st.write(f"Precision: {report_lr['1']['precision']:.2f}")
        st.write(f"Recall: {report_lr['1']['recall']:.2f}")
    except Exception as e:
        st.error("Error in Logistic Regression: " + str(e))
        st.write(traceback.format_exc())
    
    # Decision Tree
    try:
        st.subheader("Decision Tree")
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(x_train, y_train)
        y_pred_dt = dt.predict(x_test)
        conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
        accuracy_dt = accuracy_score(y_test, y_pred_dt)
        report_dt = classification_report(y_test, y_pred_dt, output_dict=True)
        metrics_dict['Decision Tree'] = {
            'Accuracy': accuracy_dt,
            'F1 Score': report_dt['1']['f1-score'],
            'Precision': report_dt['1']['precision'],
            'Recall': report_dt['1']['recall']
        }
        predictions['Decision Tree'] = y_pred_dt
        st.write("Confusion Matrix:")
        st.write(conf_matrix_dt)
        st.write(f"Accuracy: {accuracy_dt:.2f}")
        st.write(f"F1 Score: {report_dt['1']['f1-score']:.2f}")
        st.write(f"Precision: {report_dt['1']['precision']:.2f}")
        st.write(f"Recall: {report_dt['1']['recall']:.2f}")
    except Exception as e:
        st.error("Error in Decision Tree: " + str(e))
        st.write(traceback.format_exc())
    
    # Random Forest
    try:
        st.subheader("Random Forest")
        rf = RandomForestClassifier(n_estimators=500, random_state=42)
        rf.fit(x_train, y_train)
        y_pred_rf = rf.predict(x_test)
        conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
        metrics_dict['Random Forest'] = {
            'Accuracy': accuracy_rf,
            'F1 Score': report_rf['1']['f1-score'],
            'Precision': report_rf['1']['precision'],
            'Recall': report_rf['1']['recall']
        }
        predictions['Random Forest'] = y_pred_rf
        st.write("Confusion Matrix:")
        st.write(conf_matrix_rf)
        st.write(f"Accuracy: {accuracy_rf:.2f}")
        st.write(f"F1 Score: {report_rf['1']['f1-score']:.2f}")
        st.write(f"Precision: {report_rf['1']['precision']:.2f}")
        st.write(f"Recall: {report_rf['1']['recall']:.2f}")
    except Exception as e:
        st.error("Error in Random Forest: " + str(e))
        st.write(traceback.format_exc())
    
    # XGBoost
    try:
        st.subheader("XGBoost")
        # Suppress warning for label encoding
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb.fit(x_train, y_train)
        y_pred_xgb = xgb.predict(x_test)
        conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
        metrics_dict['XGBoost'] = {
            'Accuracy': accuracy_xgb,
            'F1 Score': report_xgb['1']['f1-score'],
            'Precision': report_xgb['1']['precision'],
            'Recall': report_xgb['1']['recall']
        }
        predictions['XGBoost'] = y_pred_xgb
        st.write("Confusion Matrix:")
        st.write(conf_matrix_xgb)
        st.write(f"Accuracy: {accuracy_xgb:.2f}")
        st.write(f"F1 Score: {report_xgb['1']['f1-score']:.2f}")
        st.write(f"Precision: {report_xgb['1']['precision']:.2f}")
        st.write(f"Recall: {report_xgb['1']['recall']:.2f}")
    except Exception as e:
        st.error("Error in XGBoost: " + str(e))
        st.write(traceback.format_exc())
    
    # Model Comparison
    try:
        st.subheader("Model Comparison Metrics")
        metric_df = pd.DataFrame({
            'Model': list(metrics_dict.keys()),
            'Accuracy': [metrics_dict[m]['Accuracy'] for m in metrics_dict],
            'F1 Score': [metrics_dict[m]['F1 Score'] for m in metrics_dict],
            'Precision': [metrics_dict[m]['Precision'] for m in metrics_dict],
            'Recall': [metrics_dict[m]['Recall'] for m in metrics_dict]
        })
        st.dataframe(metric_df)
        
        # Plot each metric comparison
        for metric in ['Accuracy', 'F1 Score', 'Precision', 'Recall']:
            st.subheader(f"Comparison of {metric}")
            fig, ax = plt.subplots(figsize=(6,3))
            sns.lineplot(x=metric_df['Model'], y=metric_df[metric], marker='o', ax=ax)
            ax.set_title(f"{metric} Comparison Across Models")
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.error("Error during model comparison: " + str(e))
    
    # Display Predictions Comparison
    try:
        st.subheader("Predictions Comparison")
        pred_df = pd.DataFrame({
            'Logistic Regression': predictions.get('Logistic Regression', []),
            'Decision Tree': predictions.get('Decision Tree', []),
            'Random Forest': predictions.get('Random Forest', []),
            'XGBoost': predictions.get('XGBoost', []),
            'Actual': y_test.values
        })
        st.dataframe(pred_df)
    except Exception as e:
        st.error("Error displaying predictions: " + str(e))

def main():
    st.title("Heart Disease Analysis & Model Comparison App")
    st.markdown("""
    This application performs an analysis on the heart disease dataset.  
    Use the sidebar to navigate between Data Overview, Exploratory Data Analysis, and Model Training & Comparison.
    """)
    
    # Sidebar: file uploader and navigation
    uploaded_file = st.sidebar.file_uploader("Upload heart_disease.csv", type=["csv"])
    section = st.sidebar.radio("Select Section", ("Data Overview", "Exploratory Data Analysis", "Model Training & Comparison"))
    
    # Attempt to load data
    df = load_data(file_path="heart_disease.csv", uploaded_file=uploaded_file)
    if df is None:
        st.error("Data could not be loaded. Please upload a valid CSV file.")
        return

    # Render selected section
    if section == "Data Overview":
        display_data_overview(df)
    elif section == "Exploratory Data Analysis":
        perform_eda(df)
    elif section == "Model Training & Comparison":
        train_and_compare_models(df)
    else:
        st.error("Unknown section selected.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred: " + str(e))
        st.write(traceback.format_exc())





