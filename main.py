from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model once at startup
model = joblib.load("dyslexia_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from frontend (including gender, age, and game rounds)
        data = request.get_json()
        gender = data.get('gender')
        age = data.get('age')
        round_data = data.get('rounds', [])

        # Construct feature list for prediction
        features = [gender, age]  # Gender (0/1), Age (int)

        for round_info in round_data:
            features.extend([
                round_info.get('Clicks', 0),
                round_info.get('Hits', 0),
                round_info.get('Misses', 0),
                round_info.get('Score', 0),
                round_info.get('Accuracy', 0),
                round_info.get('Missrate', 0)
            ])

        # Ensure correct number of features (26 expected: Gender + Age + 24 game-related features)
        if len(features) != 26:
            return jsonify({"error": f"Incorrect number of features. Expected 26, got {len(features)}"})

        # Predict dyslexia (1 = detected, 0 = not detected)
        prediction = model.predict([features])[0]

        result = "Dyslexia Detected" if prediction == 1 else "No Dyslexia Detected"
        details = (
            "The prediction suggests dyslexia due to low accuracy and high miss rates in multiple rounds, "
            "indicating difficulty in visual processing and letter recognition."
            if prediction == 1 else
            "The results suggest no dyslexia, as accuracy and hit rates are within the normal range, "
            "indicating good response time and cognitive processing."
        )

        return jsonify({"result": result, "details": details})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
