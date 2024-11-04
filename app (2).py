from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
with open('HousePrice/house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data from the request
    square_footage = float(request.form['square_footage'])
    num_bedrooms = int(request.form['num_bedrooms'])
    num_bathrooms = int(request.form['num_bathrooms'])
    location = request.form['location']
    
    # One-hot encode the location input
    location_suburban = 1 if location == 'Suburban' else 0
    location_urban = 1 if location == 'Urban' else 0
    
    # Create an array of the input features
    input_features = np.array([[square_footage, num_bedrooms, num_bathrooms, location_suburban, location_urban]])
    
    # Make the prediction using the model
    predicted_price = model.predict(input_features)[0]
    
    # Render the result on the HTML page
    return render_template('index.html', prediction_text=f'Predicted House Price: ${predicted_price:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)
