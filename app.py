# Import necessary libraries
import numpy as np  # For numerical operations and arrays
from flask import Flask, request, jsonify, render_template  # For creating web application, handling requests, and rendering templates
import pickle  # For loading the machine learning model

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained machine learning model from a file (model.pkl)
# This model is used to make predictions
model = pickle.load(open('model.pkl', 'rb'))

# Define the home route (the default page of the web app)
@app.route('/')
def home():
    # Renders the index.html file when accessing the home page
    return render_template('index.html')

# Define the predict route, which will process the form input and return predictions
@app.route('/predict', methods=['POST'])
def predict():
    '''
    This function takes input from the HTML form (via POST method),
    processes the input, passes it to the model, and then renders
    the prediction result back on the HTML page.
    '''
    # Get input data from the form, convert it to integers
    int_features = [int(x) for x in request.form.values()]

    # Convert the input into a NumPy array, as the model expects this format
    final_features = [np.array(int_features)]

    # Make a prediction using the loaded model
    prediction = model.predict(final_features)

    # Round off the prediction result to two decimal places
    output = round(prediction[0], 2)

    # Render the result on the HTML page (index.html)
    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

# Run the app only if this script is executed directly (useful during development)
if __name__ == "__main__":
    # Start the Flask app in debug mode (helps to see errors)
    app.run(debug=True)
