from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('linreg.pkl', 'rb') as f:
    model = pickle.load(f)

# Load column names (for one-hot / correct ordering)
with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)

# Optional: load scaler if used during training
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data as dictionary
        form_data = request.form.to_dict()
        
        # Prepare input array
        input_data = np.zeros(len(columns))  # initialize with zeros

        for i, col in enumerate(columns):
            # Check if column corresponds to a categorical value
            if col in form_data:
                # One-hot: mark 1 if form value matches column
                input_data[i] = 1
            else:
                # Otherwise try numeric conversion
                try:
                    input_data[i] = float(form_data.get(col, 0))
                except ValueError:
                    input_data[i] = 0

        # Scale numeric features if scaler is available
        if scaler:
            input_data = scaler.transform([input_data])[0]

        print("Input array sent to model:", input_data)  # debug: check correctness

        # Make prediction
        prediction = model.predict([input_data])
        prediction = np.round(prediction, 2).item()

        return render_template('index.html', prediction_text=f'Prediction: {prediction}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)