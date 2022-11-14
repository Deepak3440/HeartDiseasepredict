from flask import Flask , render_template,request
import joblib
import numpy as np


model = joblib.load('model.sav')

app = Flask(__name__)


# we have written /detect in here as I have changed in the index1.html file in href I have written the /detect
@app.route('/detect',methods=['POST','GET'])
def index():
    if request.method == "POST":
        user_input = str(request.form['user_input'])
        a = str(request.form['user_input1'])
        b = str(request.form['user_input2'])
        c = str(request.form['user_input3'])
        d= str(request.form['user_input4'])
        e = str(request.form['user_input5'])
        f = str(request.form['user_input6'])
        g= str(request.form['user_input7'])
        h = str(request.form['user_input8'])
        i = str(request.form['user_input9'])
        j = str(request.form['user_input10'])
        k = str(request.form['user_input11'])
        l = str(request.form['user_input12'])

        #print(user_input)
        l1=[user_input,a,b,c,d,e,f,g,h,i,j,k,l]
        inputis = tuple(l1)
        print(inputis)
        input_data_as_numpy_array = np.asarray(inputis)

        # reshape the numpy array as we are predicting for only on instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        result=model.predict(input_data_reshaped)
        print(result)
        if (result[0] == 0):
            data='The Person does not have a Heart Disease'
            print('The Person does not have a Heart Disease')
        else:
            data='The Person has Heart Disease'
            print('The Person has Heart Disease')
        return render_template('index.html',data=data)
    else:
        return render_template('index.html')

@app.route('/')
def home():
    return render_template('index1.html')

if __name__=='__main__':
    app.run(debug=True)