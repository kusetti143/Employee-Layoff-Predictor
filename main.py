import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import subprocess

# Set Streamlit page config with a custom theme color
st.set_page_config(page_title="Employee Layoff Predictor", layout="centered", page_icon="👥")

# Custom CSS for colors and style
st.markdown(
    """
    <style>
    .main-header {
        font-size:2.5rem;
        font-weight:700;
        color:#fff;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align:center;
        box-shadow: 0 4px 12px rgba(30,60,114,0.15);
    }
    .stButton>button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
    }
    .sidebar-header {
        color: #2a5298;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-header">Employee Layoff Predictor</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-header">Options</div>', unsafe_allow_html=True)

# Add About option in sidebar
about = st.sidebar.checkbox("About this app")
if about:
    st.sidebar.markdown(
        """
        **Employee Layoff Predictor**
        
        This app uses machine learning to predict the likelihood of employee layoffs based on HR data. 
        - Upload a CSV or enter details manually
        - View predictions, probabilities, and visual summaries
        - Download results for further analysis
        """
    )

# Load model and preprocessing objects
def load_objects():
    model = joblib.load(os.path.join('model_train', 'model.pkl'))
    scaler = joblib.load(os.path.join('model_train', 'scaler.pkl'))
    columns = joblib.load(os.path.join('model_train', 'model_columns.pkl'))
    return model, scaler, columns

model, scaler, model_columns = load_objects()

input_mode = st.sidebar.radio("Select input mode:", ("Upload CSV", "Manual Entry"))

# Helper for prediction
def preprocess_input(df, scaler, model_columns):
    # One-hot encode all object columns except EmployeeID if present
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'EmployeeID']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    # Add missing columns
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    # Drop extra columns
    df = df[model_columns]
    # Dynamically scale numeric columns if present
    numeric_cols = [col for col in ['Tenure', 'Salary', 'PerformanceScore'] if col in df.columns]
    if numeric_cols:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df

# Project description and instructions
st.markdown(
    """
    <div style='background: #f0f4f8; border-radius: 10px; padding: 1.2rem; margin-bottom: 1.5rem;'>
    <h3 style='color:#1e3c72;'>Welcome to the Employee Layoff Predictor!</h3>
    <ul>
        <li>Upload a CSV file or enter employee details manually to predict layoff risk.</li>
        <li>View predictions, probabilities, and a visual summary of results.</li>
        <li>Download your results for further analysis.</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Model Retraining")
retrain_file = st.sidebar.file_uploader("Upload new training CSV", type=["csv"], key="retrain")
if retrain_file is not None:
    with open("TrainDatasets/user_uploaded.csv", "wb") as f:
        f.write(retrain_file.getbuffer())
    st.sidebar.success("File uploaded. Ready to retrain.")
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Retraining model... This may take a while."):
            result = subprocess.run(
                ["python", "train_model.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.sidebar.success("Model retrained successfully! The app will now use the new model.")
            else:
                st.sidebar.error(f"Retraining failed:\n{result.stderr}")

if input_mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload Employee CSV", type=["csv"])
    if uploaded_file:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.stop()
        # Data validation
        missing_cols = [col for col in ['EmployeeID', 'Layoff', 'Tenure', 'Salary', 'PerformanceScore'] if col in input_df.columns and input_df[col].isnull().any()]
        if missing_cols:
            st.error(f"Uploaded file contains missing values in: {', '.join(missing_cols)}. Please clean your data and try again.")
            st.dataframe(input_df[input_df[missing_cols].isnull().any(axis=1)])
            st.stop()
        # Check for invalid (non-numeric) values in numeric columns
        for col in ['EmployeeID', 'Tenure', 'Salary', 'PerformanceScore']:
            if col in input_df.columns and not pd.api.types.is_numeric_dtype(input_df[col]):
                st.error(f"Column '{col}' must be numeric. Please fix your data and try again.")
                st.stop()
        # Proceed with prediction if all checks pass
        processed_df = preprocess_input(input_df, scaler, model_columns)
        preds = model.predict(processed_df)
        probs = model.predict_proba(processed_df)[:, 1]
        input_df['Layoff Prediction'] = np.where(preds == 1, 'Yes', 'No')
        input_df['Probability'] = probs.round(3)
        # Summary card
        n_total = len(input_df)
        n_laid_off = (input_df['Layoff Prediction'] == 'Yes').sum()
        n_retained = (input_df['Layoff Prediction'] == 'No').sum()
        st.markdown(f"""
        <div style='display:flex;gap:2rem;margin-bottom:1rem;'>
            <div style='background:#e3eafc;padding:1rem 2rem;border-radius:8px;'><b>Total Employees:</b><br><span style='font-size:1.5rem;color:#1e3c72;'>{n_total}</span></div>
            <div style='background:#ffebee;padding:1rem 2rem;border-radius:8px;'><b>Laid Off:</b><br><span style='font-size:1.5rem;color:#e53935;'>{n_laid_off}</span></div>
            <div style='background:#e8f5e9;padding:1rem 2rem;border-radius:8px;'><b>Retained:</b><br><span style='font-size:1.5rem;color:#43a047;'>{n_retained}</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            '<div style="display:flex;gap:1.5rem;align-items:center;margin-bottom:1rem;">'
            '<span style="display:inline-block;width:22px;height:22px;background:#e53935;border-radius:4px;margin-right:0.5rem;"></span>'
            '<span style="font-size:1rem;">Laid Off</span>'
            '<span style="display:inline-block;width:22px;height:22px;background:#43a047;border-radius:4px;margin-left:2rem;margin-right:0.5rem;"></span>'
            '<span style="font-size:1rem;">Retained</span>'
            '</div>', unsafe_allow_html=True)
        st.dataframe(input_df.style.applymap(lambda x: 'background-color: #e57373' if x == 'Yes' else 'background-color: #a5d6a7' if x == 'No' else '', subset=['Layoff Prediction']))
        st.download_button(
            label="Download Results as CSV",
            data=input_df.to_csv(index=False).encode('utf-8'),
            file_name='layoff_predictions.csv',
            mime='text/csv',
        )
        st.write("### Layoff Distribution")
        pie_data = input_df['Layoff Prediction'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=['#e53935', '#43a047'], startangle=90, counterclock=False)
        ax.axis('equal')
        st.pyplot(fig)
        # --- Visualization Enhancements ---
        st.write("### Layoffs by Department")
        if 'Department' in input_df.columns:
            dept_counts = input_df.groupby('Department')['Layoff Prediction'].value_counts().unstack().fillna(0)
            fig, ax = plt.subplots()
            dept_counts.plot(kind='bar', stacked=True, color=['#43a047', '#e53935'], ax=ax)
            ax.set_ylabel('Count')
            ax.set_title('Layoffs by Department')
            st.pyplot(fig)
        st.write("### Layoffs by Job Role")
        if 'JobRole' in input_df.columns:
            job_counts = input_df.groupby('JobRole')['Layoff Prediction'].value_counts().unstack().fillna(0)
            fig, ax = plt.subplots(figsize=(8, 4))
            job_counts.plot(kind='bar', stacked=True, color=['#43a047', '#e53935'], ax=ax)
            ax.set_ylabel('Count')
            ax.set_title('Layoffs by Job Role')
            st.pyplot(fig)
        # (Confusion Matrix feature removed)
else:
    st.write("### Enter Employee Details")
    emp_id = st.number_input("EmployeeID", min_value=1, step=1)
    tenure = st.number_input("Tenure (years)", min_value=0, step=1)
    salary = st.number_input("Salary", min_value=0, step=1000)
    perf = st.slider("Performance Score", 1, 10, 5)
    dept = st.selectbox("Department", ["Sales", "Engineering", "HR", "Finance", "Marketing", "Other"])
    jobrole = st.text_input("Job Role", "Employee")
    if st.button("Predict Layoff"):
        row = pd.DataFrame({
            'EmployeeID': [emp_id],
            'Tenure': [tenure],
            'Salary': [salary],
            'PerformanceScore': [perf],
            'Department': [dept],
            'JobRole': [jobrole]
        })
        processed = preprocess_input(row, scaler, model_columns)
        pred = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0, 1]
        if pred == 1:
            st.markdown(f'<div style="background-color:#e53935;padding:1rem;border-radius:8px;color:white;font-size:1.2rem;font-weight:600;">Prediction: <b>Yes</b> (Probability: {prob:.2f})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color:#43a047;padding:1rem;border-radius:8px;color:white;font-size:1.2rem;font-weight:600;">Prediction: <b>No</b> (Probability: {prob:.2f})</div>', unsafe_allow_html=True)

 
