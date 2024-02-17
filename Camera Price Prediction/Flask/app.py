import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    input_feature = [float(x) for x in request.form.values() if x]
    input_feature=[np.array(input_feature)]

    print(input_feature)
    
    names = ['Max resolution', 'Low resolution','Effective pixels', 'Zoom wide (W)', 'Zoom tele (T)', 'Normal focus range', 'Macro focus range', 'Storage included', 'Weight (inc. batteries)', 'Dimensions']
    data = pd.DataFrame(input_feature,columns=names)

    print(data)
    prediction=model.predict(data)
    output = prediction[0]
    return render_template('output.html', prediction_text='Price of Camera with specified Specifications is: {}'.format(output))
   
if __name__ == "__main__":
    app.run(debug=True)