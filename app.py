from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        message = request.form.get("message")

        if message:
            data = vectorizer.transform([message])
            result = model.predict(data)[0]

            prediction = "Spam" if result == 1 else "Not Spam"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



