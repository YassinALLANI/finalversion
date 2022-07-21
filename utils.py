# Utils for preprocessing data etc
from flask import Flask
app = Flask(__name__)
import tensorflow as tf
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

base_classes = ['tooth']

classes_and_models = {
    "model_1": {
        "classes": base_classes,
        "model_name": "detectionDental"  # change to be your model name
    },
}

import googleapiclient.discovery

def predict_json(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        region (str): regional endpoint to use; set to None for ml.googleapis.com
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)
    service = googleapiclient.discovery.build(
        'ml', 'v1', client_options=client_options)
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, rescale=False):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).
    """
    # Decode it into a tensor
    #   img = tf.io.decode_image(filename) # no channels=3 means model will break for some PNG's (4 channels)
    img = tf.io.decode_image(filename, channels=3)  # make sure there's 3 colour channels (for PNG's)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    # Rescale the image (get all values between 0 and 1)
    if rescale:
        return img / 255.
    else:
        return img


# def update_logger(image, model_used, pred_class, pred_conf, correct=False, user_label=None):
#     """
#     Function for tracking feedback given in app, updates and reutrns
#     logger dictionary.
#     """
#     logger = {
#         "image": image,
#         "model_used": model_used,
#         "pred_class": pred_class,
#         "pred_conf": pred_conf,
#         "correct": correct,
#         "user_label": user_label
#     }
#     return logger

# Setup environment credentials (you'll need to change these)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "teethdetection-354307-48d18c241917.json" # change for your GCP key
PROJECT = "Teethdetection" # change for your GCP project
REGION = "us-central1" # change for your GCP region (where your model is hosted)

def make_prediction(image, model, class_names):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.
    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    image = load_and_prep_image(image)
    # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    # image = tf.expand_dims(image, axis=0)
    preds = predict_json(project=PROJECT,
                         region=REGION,
                         model=model,
                         instances=image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    return image, pred_class, pred_conf

CLASSES = classes_and_models["model_1"]["classes"]
MODEL = classes_and_models["model_1"]["model_name"]

@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
            import base64
            # with open(file, "rb") as image_file:
            file2 = base64.urlsafe_b64encode(file.read()).decode()
            a,b,c = make_prediction(file, model=MODEL, class_names=CLASSES)

#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         # print('upload_video filename: ' + filename)
#         flash('Video successfully uploaded and displayed below')
#         print("hneeeeeeeeeeeeee",filename)
#         p = os.path.join('data/video/afterDetection', filename)
#         # subprocess.run(['python', 'detect_video.py', '--weights', './checkpoints/yolov4-416', '--size', '416', '--model', 'yolov4','--video', os.path.join(app.config['UPLOAD_FOLDER'], filename), '--output', p, '--crop', '--count'])
#         subprocess.run(
#             ['python', 'detect.py', '--weights', './checkpoints/yolov4-416', '--size', '416', '--model', 'yolov4',
#              '--images', os.path.join(app.config['UPLOAD_FOLDER'], filename), '--output', p, '--crop', '--count'])
#
    return b

if __name__ == "__main__":
    app.run()