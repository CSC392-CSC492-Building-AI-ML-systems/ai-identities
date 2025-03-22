from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/identify-model', methods=['POST'])
def identify_model():
    data = request.json
    api_key = data.get('api_key')
    provider = data.get('provider')

    if not api_key or not provider:
        return jsonify({"error": "Missing API key or provider"}), 400

    # Dummy response simulating model identification
    model_info = {
        "provider": provider,
        "model": "Sample AI Model v1.0",
        "confidence": "95%"
    }

    return jsonify(model_info)

if __name__ == '__main__':
    app.run(debug=True)
