import numpy as np
from flask import Flask, request, render_template,jsonify
import pickle
import pandas as pd
import traceback

app = Flask(__name__)
model = pickle.load(open('weight_pred_model.pkl', 'rb'))


@app.route('/getprediction',methods=['POST'])
def getprediction(): 
    if model:
        try:
            json_ = request.json
            print(json_["height"])            
            final_input = [np.array(json_["height"])]
            print(final_input)
            prediction = model.predict([final_input])            
            return jsonify(weight = str(prediction))
        except:
            return jsonify(trace = traceback.format_exc())
            

if __name__ == "__main__":
    app.run(debug=True, port=1234)
   
    