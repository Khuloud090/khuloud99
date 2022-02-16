from flask import Flask
import pickle
import pandas as pd

app = Flask(__name__)

@app.route("/test")
def run_model():
    titanic_test = pd.read_csv('test.csv')
  
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X_test = pd.get_dummies(titanic_test[features])

    with open('model.pkl' , 'rb') as f:
        model = pickle.load(f)
    predictions = pd.DataFrame(model.predict(X_test))

    return predictions.to_json()

@app.route("/hello")
def hello():
    return "Hello World"

    
if __name__ == "__main__":
    app.run(debug=True)

# to run -> `python -m flask run`