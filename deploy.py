from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Check if the model file exists before loading it
model_file = 'savemodel.joblib'  # Ensure this matches the saved model file name
if os.path.exists(model_file):
    Model = joblib.load(model_file)
else:
    raise FileNotFoundError(f"The model file '{model_file}' does not exist.")

@app.route('/')
def home():
    result = ''
    return render_template('index.html', result=result)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data and validate as floats
    sepal_length_str = request.form['sepal_length']
    sepal_width_str = request.form['sepal_width']
    petal_length_str = request.form['petal_length']
    petal_width_str = request.form['petal_width']

    # Check if any input is empty or not a valid float
    if '' in [sepal_length_str, sepal_width_str, petal_length_str, petal_width_str]:
        result = 'Please enter valid numeric values for all fields.'
    else:
        try:
            # Convert form data to float
            sepal_length = float(sepal_length_str)
            sepal_width = float(sepal_width_str)
            petal_length = float(petal_length_str)
            petal_width = float(petal_width_str)

            # Perform prediction
            result = Model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        except ValueError:
            result = 'Invalid input. Please enter valid numeric values.'

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
