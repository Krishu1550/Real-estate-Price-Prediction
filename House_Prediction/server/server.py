
from flask import Flask,request,jsonify
import ulits
app= Flask(__name__)

@app.route('/get_location_names', methods=['GET'])
def get_location_name():
    response = jsonify({
        'locations': ulits.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_home_price', methods=['GET','POST'])
def get_prediction():
    if request.method == 'POST':
        sqft = float(request.form['total_sqft'])
        location = request.form['location']
        bhk = int(request.form['bhk'])
        bath = int(request.form['bath'])
        balcony= int(request.form['balcony'])
        response = jsonify({
            'estimated_price': ulits.predict_model(location,sqft,bath,balcony,bhk)
        })
    else:
        response= "It is get method"
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__" :
    print("Starting Flask")
    ulits.load_data()
    print(ulits.predict_model('Whitefield',1000,2,3,2))
    print("run")
    app.run('0.0.0.0',port=80,debug=True)
