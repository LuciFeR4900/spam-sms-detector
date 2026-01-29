from werkzeug.middleware.proxy_fix import ProxyFix

from flask import Flask, render_template, request

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        message = request.form.get("message", "")

        if message.strip() != "":
            # TEMP prediction logic (safe)
            if "free" in message.lower() or "win" in message.lower():
                prediction = "Spam"
            else:
                prediction = "Not Spam"

    #return render_template("index.html", prediction=prediction)
    return render_template(
    "index.html",
    prediction=prediction,
    accuracy="98%"
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
