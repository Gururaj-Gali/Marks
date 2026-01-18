from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])
    prediction = model.predict(np.array([[hours]]))
    return jsonify({"predicted_marks": round(prediction[0], 2)})

if __name__ == "__main__":
    app.run(debug=True)
