from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__,template_folder='templates')

# Load the model and scaler
model = pickle.load(open('rf_classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Prediction function
def predict(model, scaler, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes,
            totChol, sysBP, diaBP, BMI, heartRate, glucose):
    # Encode categorical variables
    male_encoded = 1 if male.lower() == "male" else 0
    currentSmoker_encoded = 1 if currentSmoker.lower() == "yes" else 0
    BPMeds_encoded = 1 if BPMeds.lower() == "yes" else 0
    prevalentStroke_encoded = 1 if prevalentStroke.lower() == "yes" else 0
    prevalentHyp_encoded = 1 if prevalentHyp.lower() == "yes" else 0
    diabetes_encoded = 1 if diabetes.lower() == "yes" else 0

    # Prepare features array
    features = np.array([[male_encoded, age, currentSmoker_encoded, cigsPerDay, BPMeds_encoded, prevalentStroke_encoded,
                          prevalentHyp_encoded, diabetes_encoded, totChol, sysBP, diaBP, BMI, heartRate, glucose]])

    # Scale the features
    scaled_features = scaler.transform(features)

    # Predict using the model
    result = model.predict(scaled_features)

    return result[0]

# Routes
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if request.method == 'POST':
        try:
            # Extract data from form
            male = request.form['male']
            age = int(request.form['age'])
            currentSmoker = request.form['currentSmoker']
            cigsPerDay = float(request.form['cigsPerDay'])
            BPMeds = request.form['BPMeds']
            prevalentStroke = request.form['prevalentStroke']
            prevalentHyp = request.form['prevalentHyp']
            diabetes = request.form['diabetes']
            totChol = float(request.form['totChol'])
            sysBP = float(request.form['sysBP'])
            diaBP = float(request.form['diaBP'])
            BMI = float(request.form['BMI'])
            heartRate = float(request.form['heartRate'])
            glucose = float(request.form['glucose'])

            # Call the predict function
            prediction = predict(model, scaler, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes,
                                  totChol, sysBP, diaBP, BMI, heartRate, glucose)
            
            # Display appropriate prediction message
            prediction_text = "The Patient has Heart Disease" if prediction == 1 else "The Patient has No Heart Disease"
            return render_template('index.html', prediction=prediction_text)

        except Exception as e:
            return render_template('index.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
