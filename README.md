# Employee Layoff Predictor

A machine learning web app to predict employee layoffs based on HR data.

## Features
- Data preprocessing and model training (Logistic Regression, Random Forest, XGBoost)
- Class imbalance handling with SMOTE and class weights
- Hyperparameter tuning for Random Forest and XGBoost
- Model evaluation and automatic best model selection
- All models and preprocessing objects saved in `model_train/`
- Streamlit web app for predictions
- CSV upload or manual entry
- Color-coded results and pie chart for layoff distribution
- Bar charts for layoffs by Department and Job Role
- Data validation and user-friendly error messages for uploads
- Easy dataset switching for training

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the model:**
   - By default, trains on `TrainDatasets/Emp_Train_2.csv`.
   - To use a different dataset, edit the `TRAIN_FILE` variable at the top of `train_model.py`.
   ```python
   TRAIN_FILE = 'TrainDatasets/Emp_Train_2.csv'  # Change this to use a different dataset
   ```
   - Then run:
   ```bash
   python train_model.py
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run main.py
   ```

## Usage
- **Upload a CSV** or **enter employee details manually** in the app.
- View layoff prediction and probability.
- See a color-coded table, pie chart, and bar charts of predictions.
- Download results as CSV.
- All models and preprocessing objects are saved in `model_train/`.

## Data Format
- Input CSV should have columns:
  - EmployeeID, Tenure, Salary, PerformanceScore, Department, JobRole

## Files
- `train_model.py`: Data processing, model training (with SMOTE, class weights, hyperparameter tuning), and saving.
- `main.py`: Streamlit web app (colorful UI, pie chart, bar charts, CSV/manual input, data validation).
- `model_train/`: Folder containing all trained models and preprocessing objects (`model.pkl`, `scaler.pkl`, `model_columns.pkl`, etc.).
- `requirements.txt`: All dependencies.

## Notes
- For best results, ensure your training data is clean and formatted as described.
- You can add new datasets to the `TrainDatasets/` folder and update `TRAIN_FILE` to train on them.
- The app uses the best model (by F1-score) for predictions.
- The confusion matrix and ROC curve are not shown in the current version. 