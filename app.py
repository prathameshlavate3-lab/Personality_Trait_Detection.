from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("personality_model.joblib")

TRAITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism"
]

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("text", "").strip()

        if user_text:
            scores = model.predict([user_text])[0]
            scores = [max(0, min(100, round(float(s)))) for s in scores]
            results = list(zip(TRAITS, scores))

    return render_template(
        "index.html",
        results=results,
        user_text=user_text
    )

if __name__ == "__main__":
    app.run(debug=True)