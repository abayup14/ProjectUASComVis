from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_sentiment', methods=['POST'])
def get_sentiment():
    return jsonify({"sentiment": "neutral",
                    "sentiment_conf": 0.6,
                    "emotion": "happy",
                    "emotion_conf": 0.8
                    })


if __name__ == '__main__':
    app.run()