import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)
model_placement = pickle.load(open('placement.pkl', 'rb'))
model_salary=pickle.load(open('salary.pkl', 'rb'))

#['ssc_p',
# 'hsc_p',
# 'degree_p',
# 'etest_p',
# 'mba_p',
# 'gender_M',
# 'degree_t_Sci&Tech',
# 'workex_Yes']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model_placement.predict(final_features)

    out={'Not Placed':'No', 'Placed':'Yes'}

    output=out[prediction[0]]
    if output=='Yes':
        salary = model_salary.predict(final_features)
        salary=round(salary[0],-5)*12/100000
        return render_template('index.html', prediction_text=' {}'.format(output)+'     Approx. Salary: {}LPA'.format(salary))

    if output == 'No':
        return render_template('index.html', prediction_text=' {}'.format(output))

@app.route('/api',methods=['GET'])
def api():
    dic=request.args.to_dict()
    print(dic)
    int_features = [float(x) for x in dic.values()]
    final_features = [np.array(int_features)]
    prediction = model_placement.predict(final_features)

    out={'Not Placed':'No', 'Placed':'Yes'}

    output=out[prediction[0]]
    if output=='Yes':
        salary = model_salary.predict(final_features)
        salary=round(salary[0],-5)*12/100000
        return jsonify(status=output,salary=salary)

    if output == 'No':
        return jsonify(status=output,salary=0)


if __name__ == "__main__":
    app.run(debug=True)
