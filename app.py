import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import io
from PIL import Image

app = Flask(__name__)
app.config['DEBUG'] = True

model = tf.keras.models.load_model("model_comvis.h5")
alphabet = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def detect_char(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    detected_characters = []
    space_threshold = 50
    previous_x = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 10 :
            # Detect spaces by checking distance between characters
            if previous_x is not None and (x - previous_x) > space_threshold:
                detected_characters.append(' ')

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

@app.route('/get_predict', methods=['POST'])
def get_predict():
    if "image" not in request.form:
        return jsonify({"error": "No image provided"}), 400

    try:
        image_data = request.form["image"]
        image_data = image_data.split(",")[1]  # Remove data:image/png;base64 prefix
        decoded_image = base64.b64decode(image_data)
        image = np.array(Image.open(io.BytesIO(decoded_image)))

        detected_characters = detect_char(image)
        result = ""

        for char_image in detected_characters:
            if isinstance(char_image, str) and char_image == " ":
                result += " "
            else:
                char_img_rgb = cv2.cvtColor(char_image, cv2.COLOR_GRAY2RGB)
                char_img_rgb = np.expand_dims(char_img_rgb, axis=0)
                char_img_rgb = char_img_rgb.astype('float32') / 255.0

                pred = model.predict(char_img_rgb)
                best_idx = np.argmax(pred[0])
                best_char = alphabet[best_idx]
                print(best_char)
                result += best_char

        print(result)
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run()