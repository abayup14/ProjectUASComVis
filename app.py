from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['DEBUG'] = True

model = tf.keras.models.load_model("model_comvis.h5")

def detect_char(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cont = contours[0]
    imgcnt=cv2.drawContours(image, contours, -1, (0,255,0), 3)
    plt.imshow(imgcnt)
    plt.show()
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    # Initialize list to store detected characters, and set a space threshold
    detected_characters = []
    space_threshold = 50 # Adjust based on your image's spacing
    previous_x = None

    # Loop through contours to detect characters and spaces
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 25 and h > 25 :  # Filter out noise or small contours

            # Detect spaces by checking distance between characters
            if previous_x is not None and (x - previous_x) > space_threshold:
                detected_characters.append(' ')  # Add a space

            # Extract character as ROI
            char_image = binary[y:y+h, x:x+w]

            char_image_resized = cv2.resize(char_image, (56, 56))

            # Append the detected character image (as pixel data) to the list
            detected_characters.append(char_image_resized)

            # Update previous_x to the right boundary of the current character
            previous_x = x + w

    return detected_characters

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