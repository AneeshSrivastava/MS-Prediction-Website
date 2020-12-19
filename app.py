from flask import Flask, render_template,request
import pickle
import numpy as np


app = Flask(__name__, static_folder="./static")

model_LR = pickle.load(open('model_LR.pkl', 'rb'))
model_SVM = pickle.load(open('model_SVM.pkl', 'rb'))
model_rf = pickle.load(open('model_rf.pkl', 'rb'))
model_knn = pickle.load(open('model_knn.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template('Admission.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    new_data = []
    i = 0
    for x in request.form.values():
        if x == '\0':
            print("Break")
            break
        if i == 7:
            break
        x = float(x)
        new_data.append(x)
        i = i+1
    
    if x == 'LR':
        model = model_LR
        name = "LRused"
    if x =='SVM':
        model = model_SVM
        name = "SVM"
    if x =='RF':
        model = model_rf
        name = "RF"
    if x =='KNN':
        model = model_knn
        name = "KNN"

    
    # Convert to numpy array
    new_array = np.asarray(new_data)
    print("New array shape: ", new_array.shape)


    prediction = model.predict([new_array])
    np.set_printoptions(precision=2)

    if prediction > 0.5:
        return render_template('Admission.html', result='Congratulations !!', pred='You Will be Accepted', prob = "Your Chance of Admission is:{}".format (str(prediction)), model=" Name of the model used : " + str(name))
    else:
        return render_template('Admission.html', result='We are sorry but', pred='You cannot be Accepted', prob = "Your Chance of Admission is:{}".format (str(prediction)),  model=" Name of the model used : " + str(name))



if __name__ == "__main__":
    app.run(debug=True)  # This runs the application on web server
