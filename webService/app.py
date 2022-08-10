import os
from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory
import tensorflow as tf


app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

# Carica il modello e setting della size di imamgine
cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "my_model.h5")
IMAGE_SIZE = (150, 150)

# Preprocessing dell'immagine
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image /= 255.0  # normalizzazione a [0,1]
    image = 2*image-1  # normalizzazione a [-1,1]

    return image


# Load image and process
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)

    return preprocess_image(image)


# Predici e classifica l'imamgine
def classify(model, image_path):

    preprocessed_imgage = load_and_preprocess_image(image_path)
    preprocessed_imgage = tf.reshape(
        preprocessed_imgage, (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    #Le cinque classi di predizione
    prob = cnn_model.predict(preprocessed_imgage)
    label = ["No Parkinson", "Inzio della malattia",
             "Avanzamento lieve", "Avanzamento intermedio", "Utlimo Stadio della malattia"]
    xValues = prob[0]
    result = max(enumerate(xValues), key=(lambda x: x[1]))

    return label[result[0]], round((result[1] * 100), 2)



@app.route("/classify", methods=["POST", "GET"])
def upload_file():

    if request.method == "GET":
        return render_template("index.html")

    file = request.files["image"]
    if (file.filename ==""):
        return render_template("index.html")

    else:
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        #print(upload_image_path)
        file.save(upload_image_path)

        label, prob = classify(cnn_model, upload_image_path)
        #print(f"label:{label}, prob:{prob}")

    return render_template(
        "classify.html", image_file_name=file.filename, label=label, prob=prob
    )

@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Home root
@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")


if __name__ == '__main__':
    #app.debug = True
    app.run()
