import os
import threading
import concurrent.futures
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd


# -------------------------------------------------------------------
# Thread-safe model loading and prediction
# -------------------------------------------------------------------
_model_lock = threading.Lock()
_model = None
_model_loaded = threading.Event()

def _load_model():
  #  \"\"\"Load the CNN model (called once in a background thread).\"\"\"
    global _model
    print("[Thread] Loading model...")
    m = CNN.CNN(39)
    m.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location="cpu"))
    m.eval()
    # Acquire lock before assigning the shared model reference
    with _model_lock:
        _model = m
    _model_loaded.set()
    print("[Thread] Model loaded successfully.")

def get_model():
    #\"\"\"Return the model, waiting for loading if necessary.\"\"\"
    _model_loaded.wait()  # blocks until model is loaded
    return _model

def prediction(image_path):
    #\"\"\"Run prediction on an image.
    
    #Uses a thread pool executor so the inference runs on a background
    #thread, keeping the Flask request handler responsive.
    #\"\"\"
    def _infer(path):
        # Acquire the model lock for thread-safe inference
        model = get_model()
        image = Image.open(path)
        image = image.resize((224, 224))
        input_data = TF.to_tensor(image)
        input_data = input_data.view((-1, 3, 224, 224))
        
        with _model_lock:  # ensure only one inference at a time
            output = model(input_data)
        
        output = output.detach().numpy()
        index = np.argmax(output)
        return int(index)
    
    # Use a thread pool to run inference in the background
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_infer, image_path)
        return future.result()


# -------------------------------------------------------------------
# CSV data loading
# -------------------------------------------------------------------
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')


# -------------------------------------------------------------------
# Flask application
# -------------------------------------------------------------------
app = Flask(__name__)

# Start model loading in a background thread so the app boots faster
threading.Thread(target=_load_model, daemon=True).start()

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        
        pred = prediction(file_path)
        
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred, sname=supplement_name,
                               simage=supplement_image_url, buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html',
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']),
                           buy=list(supplement_info['buy link']))

@app.route('/api/predict', methods=['POST'])
def api_predict():

    try:

        if 'image' not in request.files:

            return {
                'error': 'No image uploaded'
            }

        image = request.files['image']

        if image.filename == '':

            return {
                'error': 'No image selected'
            }

        file_path = os.path.join(
            'static/uploads',
            image.filename
        )

        image.save(file_path)

        pred = prediction(file_path)

        title = disease_info['disease_name'][pred]

        description = disease_info['description'][pred]

        prevent = disease_info['Possible Steps'][pred]

        return {

            'disease': title,
            'description': description,
            'prevention': prevent

        }

    except Exception as e:

        return {
            'error': str(e)
        }
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
