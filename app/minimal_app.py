from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from Flask on Vercel!"

@app.route('/api/models')
def get_models():
    return {"models": ["Test Model 1", "Test Model 2"]}

if __name__ == '__main__':
    app.run(debug=True)

