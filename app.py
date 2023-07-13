# Import required libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Logistic Regression model
filename = 'heart-disease-prediction-logistic-regression.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = float(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = float(request.form['ca'])
        thal = request.form.get('thal')
        
        data = np.array([[float(age),sex,cp,float(trestbps),float(chol),fbs,float(restecg),float(thalach),exang,float(oldpeak),slope,float(ca),thal]])
        data = data.astype("float")
        prediction = model.predict(data)
        return render_template('result.html', prediction=prediction)
        
        

if __name__ == '__main__':
	app.run(debug=True)

