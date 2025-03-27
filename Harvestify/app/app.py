# Importing essential libraries and modules
import os
from flask import Flask, render_template, request,redirect,url_for,flash,session,jsonify
from bson import ObjectId
from flask_pymongo import PyMongo
from flask_login import LoginManager,UserMixin,login_user,logout_user,login_required,current_user
from werkzeug.security import generate_password_hash,check_password_hash
from werkzeug.utils import secure_filename
from config import Config
import tempfile

from markupsafe import Markup
import numpy as np

import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from googletrans import Translator
import re
import asyncio
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
# from tem.keras.models import load_model
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------
app = Flask(__name__)
# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']





model_path = "models/trained_model.keras"
model = load_model(model_path)

# Loading crop recommendation model

crop_recommendation_model_path = '../models/RandomForest.pkl'#models/RandomForest.pkl
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Translation function

async def translate_text(text,target_language):
    translator=Translator()
    translation=await translator.translate(text,dest=target_language)
    return translation.text



# Custom functions for calculations

def format_translated_text(text):
    # Split the text into sentences or logical chunks
    paragraphs = text.split("\n")
    
    formatted_text = ""
    for paragraph in paragraphs:
        formatted_text += f"<p>{paragraph}</p><br>"

    formatted_text=Markup(str(formatted_text))
    return formatted_text

    

# Set up the directory where you want to store the uploaded images temporarily
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed extensions (e.g., jpg, png)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def predict_image_keras(img_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))  # Match input size of model
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

    # Predict with the model
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    model_prediction = disease_classes[result_index]
    return model_prediction

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------



app.config["weather_api_key"] = "9d7cde1f6d07ec55650544be1631307e"
app.config["SECRET_KEY"] = "a3b5c6d7e8f9g0h1i2j3k4l5m6n7o8p9"
app.config["MONGO_URI"] = "mongodb+srv://HarvestifyDB:HarvestifyDB@cluster0.rtxl9.mongodb.net/farmdb?retryWrites=true&w=majority&appName=Cluster0"

mongo = PyMongo(app)
print("mongo",mongo)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# User model for Flask-Login
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data["_id"])
        self.email = user_data["email"]

@login_manager.user_loader
def load_user(user_id):
    user = mongo.db.users.find_one({"_id": user_id})
    return User(user) if user else None




# render home page


@app.route('/')
def home():
    user_id = session.get('user_id')
    if user_id:
        user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
        if user:
            return render_template('index.html', user=user)  # Display user info on the index page
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    title = "Harvestify - Login"
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = mongo.db.users.find_one({"email": email})  # Fetch user from MongoDB
        print(user)

        if user and check_password_hash(user["password"], password):
            session['user_id'] = str(user["_id"]) 
            print("auhtenticate user",user)
            # login_user(User(user))  # Log in the user
            return redirect(url_for("home"))  # Redirect to home (which renders index.html)
        else:
            flash("Invalid email or password", "danger")

    return render_template('login.html', title=title)


@app.route('/register', methods=['GET', 'POST'])
def register():
    title = "Harvestify - Register"
    if request.method == "POST":
        email = request.form.get("email")
        print(email)
        password = request.form.get("password")
        print(password)
        hashed_password = generate_password_hash(password)

        # Check if the user already exists
        existing_user = mongo.db.users.find_one({"email": email})
        if existing_user:
            print("register",existing_user)
            flash("Email already exists!", "warning")
        else:
            mongo.db.users.insert_one({"email": email, "password": hashed_password})

            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))

    return render_template('register.html', title=title)

@app.route('/dashboard')
@login_required
def dashboard():
    title = "Harvestify - Dashboard"
    return render_template('dashboard.html', title=title)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))



API_KEY = "10e7e7c3e26f198c410e125e33a6d4d5"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

@app.route('/weather-predict', methods=['GET', 'POST'])
def get_weather():
    if request.method == 'POST':
        state = request.form.get('stt')  # Matches `name="stt"` in the form
        city = request.form.get('city')  # Matches `name="city"`

        if not city:
            return render_template('weather.html', error="City is required.")

        params = {
            "q": city,
            "appid": API_KEY,
            "units": "metric"
        }

        try:
            response = requests.get(BASE_URL, params=params)
            data = response.json()

            if response.status_code != 200:
                return render_template('weather.html', error=data.get("message", "Could not fetch weather data"))

            weather_info = {
                "city": data.get("name", "N/A"),
                "temperature": data["main"].get("temp", "N/A"),
                "description": data["weather"][0].get("description", "N/A"),
                "humidity": data["main"].get("humidity", "N/A"),
                "wind_speed": data["wind"].get("speed", "N/A")
            }

            return render_template('weather.html', weather=weather_info)

        except requests.exceptions.RequestException as e:
            return render_template('weather.html', error="Error fetching weather data.")

    return render_template('weather.html')  # Initial form render

# Route for rendering the weather page
@app.route('/weather')
def weather_recommend():
    return render_template('weather.html', title="Harvestify - Crop Recommendation")


# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# ===============================================================================================

# RENDER PREDICTION PAGES


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = request.form['temperature']
        humidity = request.form['humidity']

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]

        return render_template('crop-result.html', prediction=final_prediction, title=title)


@ app.route('/fertilizer-predict', methods=['POST'])
async def fert_recommend():
    lang = request.form.get('lang', 'en') # Default to 'en' if lang is not provided
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    print(N)
    print(P)
    print(K)

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n <= 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p <= 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k <= 0:
            key = 'KHigh'
        else:
            key = "Klow"

    # response = Markup(str(fertilizer_dic[key]))

    # Check if the key exists before accessing it
 
    if key in fertilizer_dic:
        prediction = Markup(str(fertilizer_dic[key]))
    else:
        prediction = "No suggestions available for this soil type."

       

    # Step 1: Remove HTML tag

    clean_text = re.sub(r'<[^>]*>', '', prediction)
            
            # translated_text = translator.translate(clean_text, src='en', dest='es').text

    translated_text =await translate_text(clean_text,lang)
  
          

    final_text=format_translated_text(translated_text)

    print("final Text ",final_text)
   

    return render_template('disease-result.html', prediction=final_text, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
async def disease_prediction():
    lang = request.form.get('lang', 'en') # Default to 'en' if lang is not provided
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        print(f"f fie is : {file}")
        if not file or file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
             with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                filename = secure_filename(file.filename)
                temp_file_path = os.path.join(tempfile.gettempdir(), filename)
                file.save(temp_file_path)

                # Now you can pass this path to your prediction function
                prediction = predict_image_keras(temp_file_path)

                # Convert prediction to a string for display
                prediction = Markup(str(disease_dic[prediction]))

                # Step 1: Remove HTML tags

                clean_text = re.sub(r'<[^>]*>', '', prediction)

                # print("clean_text",clean_text)
                
                # translated_text = translator.translate(clean_text, src='en', dest='es').text
                translated_text =await translate_text(clean_text,lang)

                # print("translated_text",translated_text)

                final_text=format_translated_text(translated_text)
                # print("final text",final_text)
                # translated_text = translate_text(prediction,'hi') 

                # Delete the temporary file after processing
                os.remove(temp_file_path)

                # Render the template with prediction result
            
                return render_template('disease-result.html', prediction=final_text, title=title)
        
    return render_template('disease.html', title=title)

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
