import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def train_model():
    dataset = pd.read_csv('heart_cleveland_upload.csv')
    dataset = dataset.replace('?', np.nan)
    imp = IterativeImputer(random_state=0)
    imputed_dataset = pd.DataFrame(imp.fit_transform(dataset))
    X = imputed_dataset.iloc[:, :-1].values
    y = imputed_dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    joblib.dump(rfc, 'heart_disease_model.joblib')


def predict():
    rfc = joblib.load('heart_disease_model.joblib')
    age = input("Enter your age: ")
    sex = input("Enter your sex (0 for female, 1 for male): ")
    cp = input("Enter your chest pain type (0-3): ")
    trestbps = input("Enter your resting blood pressure: ")
    chol = input("Enter your cholesterol level: ")
    fbs = input("Enter your fasting blood sugar level (0 if <= 120 mg/dl, 1 if > 120 mg/dl): ")
    restecg = input("Enter your resting electrocardiographic results (0-2): ")
    thalach = input("Enter your maximum heart rate achieved: ")
    exang = input("Enter if you have exercise induced angina (0 for no, 1 for yes): ")
    oldpeak = input("Enter your ST depression induced by exercise relative to rest: ")
    slope = input("Enter the slope of the peak exercise ST segment (0-2): ")
    ca = input("Enter the number of major vessels (0-3) colored by flourosopy: ")
    thal = input("Enter your thalassemia type (0-3): ")
    user_input = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    prediction = rfc.predict(user_input)
    if prediction[0] == 0:
            print("Based on the information you provided, you are predicted to not have heart disease.")
    else:
            print("Based on the information you provided, you are predicted to have heart disease.")


if __name__ == "__main__":
    print("Welcome to the Heart Disease Predictor")
    choice = input("Enter '1' to train the model or '2' to make a prediction: ")
    if choice == '1':
        train_model()
    elif choice == '2':
        predict()
    else:
        print("Invalid choice")
