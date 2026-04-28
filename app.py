from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Numerical inputs
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
        Credit_History = float(request.form['Credit_History'])

        # Categorical (encoded values)
        Gender_Male = int(request.form['Gender_Male'])
        Married_Yes = int(request.form['Married_Yes'])

        Dependents_1 = int(request.form['Dependents_1'])
        Dependents_2 = int(request.form['Dependents_2'])
        Dependents_3 = int(request.form['Dependents_3'])

        Education_Not_Graduate = int(request.form['Education_Not_Graduate'])
        Self_Employed_Yes = int(request.form['Self_Employed_Yes'])

        Property_Area_Semiurban = int(request.form['Property_Area_Semiurban'])
        Property_Area_Urban = int(request.form['Property_Area_Urban'])

        # Feature order MUST match training dataset
        features = np.array([[
            ApplicantIncome,
            CoapplicantIncome,
            LoanAmount,
            Loan_Amount_Term,
            Credit_History,
            Gender_Male,
            Married_Yes,
            Dependents_1,
            Dependents_2,
            Dependents_3,
            Education_Not_Graduate,
            Self_Employed_Yes,
            Property_Area_Semiurban,
            Property_Area_Urban
        ]])

        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "Loan Approved ✅"
        else:
            result = "Loan Rejected ❌"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)