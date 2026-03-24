from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)
app.secret_key = "secret123"

# Load model and scaler (with fallback if not found)
has_model = False
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model.pkl')
    scaler_path = os.path.join(base_dir, 'scaler.pkl')
    
    # Check parent directory if not found in current (helpful for users running from subfolder)
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, '..', 'model.pkl')
        scaler_path = os.path.join(base_dir, '..', 'scaler.pkl')

    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    has_model = True
except Exception as e:
    print(f"Warning: model.pkl or scaler.pkl not found or failed to load. Please train and save your model to run predictions. Error: {e}")

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == '1234':
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            error = "Invalid Credentials. Please try again."
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html', has_model=has_model)

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    if not has_model:
        return render_template('index.html', prediction_text="Error: ML Model not found in the directory. Please make sure model.pkl and scaler.pkl exist.", has_model=has_model)
        
    try:
        # Get values from the form inputs in sequential order as defined in index.html
        # Using feature1 to feature22
        features = [float(request.form[f'f{i}']) for i in range(1, 23)]
        
        input_data = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)

        result = "No Parkinson's Disease Detected 🟢" if prediction[0] == 0 else "Parkinson's Disease Detected 🔴"

        return render_template('index.html', prediction_text=result, has_model=has_model, submitted=True)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error processing input: {str(e)}", has_model=has_model, submitted=True)

# 🚀 API VERSION added for Backend Roles integration
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint that accepts JSON data with 'features' key containing the 22 features.
    """
    if not has_model:
        return jsonify({'error': 'Prediction models not initialized on the server.'}), 500
        
    try:
        data = request.get_json(force=True)
        
        if 'features' not in data or len(data['features']) != 22:
            return jsonify({'error': 'Please provide exactly 22 features under the "features" key in the JSON body.'}), 400
            
        features = [float(x) for x in data['features']]
        input_data = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)
        
        # Determine the label based on your model's 0/1 output
        label = int(prediction[0])
        outcome = "Parkinson's Detected" if label == 1 else "No Parkinson's Disease"
        
        return jsonify({
            'success': True,
            'prediction_class': label,
            'prediction_message': outcome
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == "__main__":
    # Ensure templates folder exists for the user running it
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
    app.run(debug=True, port=5000)
