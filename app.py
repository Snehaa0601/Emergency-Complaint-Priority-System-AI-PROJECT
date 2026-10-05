from flask import Flask, render_template, request
import pickle


app = Flask(__name__)


# Load trained model
with open("priority_model.pkl", "rb") as file:

    model, category_encoder, priority_encoder = pickle.load(file)


# Home page
@app.route("/")
def home():

    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    # Get values from form
    category = request.form["category"]
    severity = int(request.form["severity"])
    people = int(request.form["people"])
    response = int(request.form["response"])


    # Convert category to numerical value
    category_encoded = category_encoder.transform(
        [category]
    )[0]


    # Create input for model
    input_data = [[
        category_encoded,
        severity,
        people,
        response
    ]]


    # Predict priority
    prediction = model.predict(input_data)


    # Convert number back to text
    priority = priority_encoder.inverse_transform(
        prediction
    )[0]


    return render_template(
        "index.html",
        result=priority
    )


if __name__ == "__main__":

    app.run(debug=True)
