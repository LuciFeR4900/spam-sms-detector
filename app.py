from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

data = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
data.columns = ["label", "message"]
data["label"] = data["label"].map({"ham": 0, "spam": 1})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["message"])
y = data["label"]

model = MultinomialNB()
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        print("FORM SUBMITTED")
        msg = request.form["sms"]
        msg_vector = vectorizer.transform([msg])
        prediction = model.predict(msg_vector)
        result = "🚫 Spam Message" if prediction[0] == 1 else "✅ Not Spam Message"
    #return "<h1>Flask is connected</h1>"
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
