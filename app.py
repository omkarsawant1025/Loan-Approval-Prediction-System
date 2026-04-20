from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html', columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = []

        for col in columns:
            val = request.form.get(col)

            if val is None or val == "":
                return render_template('result.html', result="Fill all fields")

            values.append(float(val))

        final_input = np.array(values).reshape(1, -1)

        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1]

        result = "Loan Approved" if prediction == 1 else "Loan Rejected"
        confidence = round(probability * 100, 2)

        return render_template('result.html', result=result, confidence=confidence)

    except Exception as e:
        return render_template('result.html', result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)