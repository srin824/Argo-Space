# load packages==============================================================
from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from transformers import pipeline
import pandas as pd
import joblib
from keras import models,layers
from PIL import Image
import pickle
import sklearn
import numpy as np
import sqlite3
import os
import json
import tensorflow as tf
from decouple import config
dtr=pickle.load(open('./Models/cropyield/dtr.pkl','rb'))
preprocessor=pickle.load(open('./Models/cropyield/preprocessor.pkl','rb'))
model1=pickle.load(open('./Models/croprecommender/model.pkl','rb'))
sc1 = pickle.load(open('./Models/croprecommender/standardscaler.pkl','rb'))
app = Flask(__name__)
app.secret_key = config("app.secret_key")
def init_sqlite_db():
    conn = sqlite3.connect('database.db')
    print("Opened database successfully")
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, password TEXT)')
    print("Table created successfully")
    conn.close()
init_sqlite_db()
#home
@app.route('/')
def home():
    if 'username' in session:
        return render_template('home.html',username=session['username'])
    else:
        return redirect(url_for('login'))
#login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            flash("Invalid login credentials. Please try again.")
            return redirect(url_for('login'))

    return render_template('login.html')
#register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password == confirm_password:
            # Generate the password hash using the default method
            hashed_password = generate_password_hash(password)

            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            conn.close()

            flash("Registration successful! Please login.")
            return redirect(url_for('login'))
        else:
            flash("Passwords do not match. Please try again.")
            return redirect(url_for('register'))

    return render_template('register.html')

#crop yield prediction
@app.route('/cropyield')
def cropyield():
    return render_template('cropyield.html')
@app.route("/predict_yield",methods=['POST'])
def predict_yield():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item  = request.form['Item']
        features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1,-1)
        return render_template('cropyield.html',prediction = prediction[0][0])

#crop prediction
@app.route('/croppredict')
def croppredict():
    return render_template("croppredict.html")

@app.route("/predict_crop",methods=['POST'])
def predict_crop():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = sc1.transform(single_pred)
    prediction = model1.predict(scaled_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0][0] in crop_dict:
        crop = crop_dict[prediction[0][0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('croppredict.html',result = result)

#weather
@app.route('/weather')
def weather():
    return render_template('weather.html')

#pages that need improvement
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/research')
def research():
    return render_template('research.html')
@app.route('/resources')
def resources():
    return render_template('resources.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/learnmore')
def learnmore():
    return render_template('learnmore.html')

@app.route('/cropMonitoring')
def cropMonitoring():
    return render_template('cropMonitoring.html')

#logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("You have been logged out.")
    return redirect(url_for('login'))

#find me
# Define a variable to store the model pipeline globally
fill_mask = None

# Function to load the model if it's not already loaded
def load_model():
    global fill_mask
    if fill_mask is None:
        fill_mask = pipeline(
            "fill-mask",
            model="recobo/agriculture-bert-uncased",
            tokenizer="recobo/agriculture-bert-uncased"
        )
        print("Model loaded!")

# Function to generate a response using the model
def generate_response(user_input):
    if "[MASK]" not in user_input:
        return "Error: The input must contain a [MASK] token for prediction."
    
    # Call load_model to ensure the model is loaded
    load_model()

    results = fill_mask(user_input)
    top_prediction = results[0]['sequence']
    
    return top_prediction

# Route to serve the HTML page
@app.route('/querybot')
def querybot():
    return render_template('querybot.html')

# API endpoint for the chatbot
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    # Get the bot response
    bot_response = generate_response(user_message)
    
    return jsonify({"response": bot_response})

#Disease prediction CNN
# Custom model architecture definition
def create_model(input_shape=(224, 224, 3), num_classes=38):  # Update num_classes based on your dataset
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(2, 2))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Update this with actual number of classes
    
    return model

# Load your pest detector model from .keras format
pest_detector_model = create_model()  # Create the model architecture
pest_detector_model.load_weights('./Models/pestdetector/plant_disease_prediction_model.h5')  # Load weights

# Load the class indices and disease information from JSON files
with open('./Models/pestdetector/class_indices.json', 'r') as file:
    class_indices = json.load(file)

with open('./Models/pestdetector/disease_info.json', 'r') as file:
    disease_info = json.load(file)

# Allowed extensions for file uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB') # Convert to RGB to ensure 3 
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.  # Normalize image
    return img_array

# Function to predict the class of the image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    print(class_indices[str(predicted_class_index)])
    # Since class_indices is a dictionary with index keys as strings, convert it
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown Class")
    return predicted_class_name

# Route for plant disease prediction page
@app.route('/diseasepredict', methods=['GET', 'POST'])
def disease_predict():
    if request.method == 'POST':
        # Check if a file is in the request
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # If the user does not select a file, the browser may submit an empty part
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('./uploads', filename)
            file.save(file_path)  # Save the uploaded image to the server

            # Predict the disease class
            predicted_class_name = predict_image_class(pest_detector_model, file_path, class_indices)
            
            # Get the corresponding disease information from the JSON file
            disease_info_text = disease_info.get(predicted_class_name, "No information available for this class.")

            return render_template('diseasepredict.html', prediction=predicted_class_name, info=disease_info_text)

    return render_template('diseasepredict.html')  # Render the form for image upload

# Ensure your uploads folder exists
if not os.path.exists('./uploads'):
    os.makedirs('./uploads')

# Step 1: Load the saved models (imputer, scaler, and random forest regressor)
imputer_soil = joblib.load('./Models/soil_moisture/imputer.pkl')
scaler_soil = joblib.load('./Models/soil_moisture/scaler.pkl')
regressor_soil = joblib.load('./Models/soil_moisture/random_forest_regressor.pkl')

@app.route('/soilWater')
def soilWater():
    # Render the input form page
    return render_template('soilWater.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Step 2: Get data from form submission
    # Split the input string into components and convert to integers
    ttime_input = str(request.form['ttime'])
    time_components = list(map(int, ttime_input.split(',')))

        # Create a datetime object from the components
    ttime = pd.to_datetime(f"{time_components[0]}-{time_components[1]}-{time_components[2]} {time_components[3]}:{time_components[4]}:{time_components[5]}")
    ttime= int(ttime.timestamp())
    pm1 = float(request.form['pm1'])
    pm2 = float(request.form['pm2'])
    pm3 = float(request.form['pm3'])
    am = float(request.form['am'])
    sm = float(request.form['sm'])
    st = float(request.form['st'])
    lum = float(request.form['lum'])

    # Step 3: Convert inputs into a NumPy array
    user_input = np.array([[ttime, pm1, pm2, pm3, am, sm, st, lum]])

    # Step 4: Data preprocessing (imputation and scaling)
    #user_input = imputer_soil.transform(user_input)  # Handle missing values if any
    user_input = scaler_soil.transform(user_input)   # Scale the data

    # Step 5: Predict using the loaded model
    prediction = regressor_soil.predict(user_input)

    # Prepare results (assuming prediction for Temperature, Humidity, and Moisture)
    result = {
        'Temperature': prediction[0][0],
        'Humidity': prediction[0][1],
        'Moisture': prediction[0][2]
    }

    # Step 6: Render the result page with the predicted output
    return render_template('result.html', result=result)
@app.route('/satellite')
def satellite():
    return render_template('satelliteimage.html')
if __name__ == '__main__':
    load_model()
    app.run(debug=True)