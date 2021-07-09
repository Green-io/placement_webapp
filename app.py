import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)
model_placement = pickle.load(open('placement.pkl', 'rb'))
model_salary=pickle.load(open('salary.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model_placement.predict(final_features)

    output = prediction[0]

    if output == 'Yes':
        salary = model_salary.predict(final_features)
        salary = round(salary[0], -5) / 10
        return render_template('index.html', prediction_text=' {}'.format(output)+'     Approx. Salary: {}LPA'.format(salary/100000))

    if output == 'No':
        return render_template('index.html', prediction_text=' {}'.format(output))

@app.route('/api',methods=['GET'])
def api():
    dic=request.args.to_dict()
    print(dic)
    int_features = [float(x) for x in dic.values()]
    final_features = [np.array(int_features)]
    prediction = model_placement.predict(final_features)

    # out={'Not Placed':'No', 'Placed':'Yes'}

    output=prediction[0]

    if output=='Yes':
        salary = model_salary.predict(final_features)
        salary=round(salary[0],-5)/10
        return jsonify(status=output,salary=salary)

    if output == 'No':
        return jsonify(status=output,salary=0)


if __name__ == "__main__":
    app.run(debug=True)
