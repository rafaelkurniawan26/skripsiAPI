import numpy as np
import flask
import xgboost as xgb
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tldextract
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import csv
from flask import request, jsonify

app = flask.Flask(__name__)

# Load the pre-trained model
bst = xgb.Booster()
bst.load_model('phishdetect.model')

# Load the scaler used during training
scalers = joblib.load('scalers.pkl') 

# List of features that require scaling or encoding
scale_features = [
    'url_length', 'number_of_dots_in_url', 'number_of_digits_in_url', 'number_of_special_char_in_url',
    'number_of_hyphens_in_url', 'number_of_underline_in_url', 'number_of_slash_in_url',
    'number_of_questionmark_in_url', 'number_of_equal_in_url', 'number_of_at_in_url', 
    'number_of_dollar_in_url', 'number_of_exclamation_in_url', 'number_of_hashtag_in_url', 
    'number_of_percent_in_url', 'domain_length', 'number_of_dots_in_domain', 'number_of_hyphens_in_domain',
    'number_of_special_characters_in_domain', 'number_of_digits_in_domain', 'number_of_subdomains', 
    'average_subdomain_length', 'average_number_of_dots_in_subdomain', 'average_number_of_hyphens_in_subdomain', 
    'number_of_special_characters_in_subdomain', 'number_of_digits_in_subdomain', 'path_length', 
    'entropy_of_url', 'entropy_of_domain'
]
def calculate_entropy(s):
    probabilities = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
    entropy = - sum([p * np.log2(p) for p in probabilities])
    return entropy

# Function to extract features from URL
def extract_features(url):
    if url.startswith('http://'):
        url = url[7:]
    elif url.startswith('https://'):
        url = url[8:]
    if url.startswith('www.'):
        url = url[4:]
    features = {}
    features['url_length'] = len(url)
    features['number_of_dots_in_url'] = url.count('.')
    features['having_repeated_digits_in_url'] = int(any(url.count(x) > 1 for x in '0123456789'))
    features['number_of_digits_in_url'] = sum(c.isdigit() for c in url)
    features['number_of_special_char_in_url'] = sum(not c.isalnum() for c in url)
    features['number_of_hyphens_in_url'] = url.count('-')
    features['number_of_underline_in_url'] = url.count('_')
    features['number_of_slash_in_url'] = url.count('/')
    features['number_of_questionmark_in_url'] = url.count('?')
    features['number_of_equal_in_url'] = url.count('=')
    features['number_of_at_in_url'] = url.count('@')
    features['number_of_dollar_in_url'] = url.count('$')
    features['number_of_exclamation_in_url'] = url.count('!')
    features['number_of_hashtag_in_url'] = url.count('#')
    features['number_of_percent_in_url'] = url.count('%')
    
    # Split URL into domain and path
    try:
        domain = url.split('/')[2]
    except IndexError:
        domain = url

    features['domain_length'] = len(domain)
    features['number_of_dots_in_domain'] = domain.count('.')
    features['number_of_hyphens_in_domain'] = domain.count('-')
    features['having_special_characters_in_domain'] = int(any(not c.isalnum() for c in domain))
    features['number_of_special_characters_in_domain'] = sum(not c.isalnum() for c in domain)
    features['having_digits_in_domain'] = int(any(c.isdigit() for c in domain))
    features['number_of_digits_in_domain'] = sum(c.isdigit() for c in domain)
    features['having_repeated_digits_in_domain'] = int(any(domain.count(x) > 1 for x in '0123456789'))
    
    # Extract subdomains
    subdomains = domain.split('.')[:-2]
    features['number_of_subdomains'] = len(subdomains)
    features['having_dot_in_subdomain'] = int(any('.' in sub for sub in subdomains))
    features['having_hyphen_in_subdomain'] = int(any('-' in sub for sub in subdomains))
    features['average_subdomain_length'] = np.mean([len(sub) for sub in subdomains]) if subdomains else 0
    features['average_number_of_dots_in_subdomain'] = np.mean([sub.count('.') for sub in subdomains]) if subdomains else 0
    features['average_number_of_hyphens_in_subdomain'] = np.mean([sub.count('-') for sub in subdomains]) if subdomains else 0
    features['having_special_characters_in_subdomain'] = int(any(any(not c.isalnum() for c in sub) for sub in subdomains))
    features['number_of_special_characters_in_subdomain'] = sum(sum(not c.isalnum() for c in sub) for sub in subdomains)
    features['having_digits_in_subdomain'] = int(any(any(c.isdigit() for c in sub) for sub in subdomains))
    features['number_of_digits_in_subdomain'] = sum(sum(c.isdigit() for c in sub) for sub in subdomains)
    features['having_repeated_digits_in_subdomain'] = int(any(any(sub.count(x) > 1 for x in '0123456789') for sub in subdomains))

    features['having_path'] = int('/' in url)
    features['path_length'] = len(url.split('/', 3)[-1]) if '/' in url else 0
    features['having_query'] = int('?' in url)
    features['having_fragment'] = int('#' in url)
    features['having_anchor'] = int('#' in url)
    features['entropy_of_url'] = calculate_entropy(url)
    features['entropy_of_domain'] = calculate_entropy(domain)

    return features

# Function to preprocess the extracted features
def preprocess(features):
    features = pd.DataFrame([features])
    features[scale_features] = scalers.transform(features[scale_features])
    return features

result=None

@app.route("/predict", methods=["POST"])
def predict():
    global result
    data = {"success": False}
    
    if flask.request.method == "POST":
        incoming = flask.request.get_json()
        url = incoming["url"]

        # Extract and preprocess features
        features = extract_features(url)
        print(f"Extracted features: {features}")  # Print the extracted features
        features = preprocess(features)

        # Predict using the pre-trained XGBoost model
        dmatrix = xgb.DMatrix(features)
        prediction = bst.predict(dmatrix)[0]
        
        data["predictions"] = []
        
        if prediction > 0.50:
            result = "URL is probably suspicious."
        else:
            result = "URL is probably safe."
        
        prediction = float(prediction)
        prediction = prediction * 100

        if result == "URL is probably safe.":

            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                login_form = soup.find('form')

                if login_form:
                    action = login_form.get('action')
                    if action:
                        domain = urlparse(url).netloc
                        action_domain = urlparse(action).netloc
                        if action_domain.startswith(domain) or domain.startswith(action_domain):
                            print("This page has a login form with URL redirect to the same domain or a subdomain.")
                        else:
                            print("This page has a login form with URL redirect to a different domain.")
                            result = "This domain doesn't show phish characteristics but the page content has a login form with URL redirect to a different domain (High phish characteristics)."
                            prediction = prediction + 50
                else:
                    print("This page has a login form with no action attribute.")
            except Exception as e:
                print(f"Error while analyzing the page: {e}")
        elif result == "URL is probably suspicious.":
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                login_form = soup.find('form')
                if login_form or soup.find_all('meta', attrs={'http-equiv':'refresh'}):
                    print("This page has a login form or redirects.")
                else:
                    print("This page doesn't have a login form and doesn't redirect.")
                    result = "This domain shows a phish characteristics, but right now the content is safe."
                    prediction = prediction - 50
            except Exception as e:
                print(f"Error while analyzing the page: {e}")
        
        r = {"result": result, "malicious percentage": f"{prediction:.2f}%", "url": url}
        data["predictions"].append(r)
        data["success"] = True

    return flask.jsonify(data)

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    feedback = data['feedback']

    if result == "URL is probably safe.":
        feedback = 0 if feedback == 'yes' else 1
    else:
        feedback = 1 if feedback == 'yes' else 0
    # Extract features from the URL
    url = data['url']
    features = extract_features(url)

    # Define the column names
    column_names = ['Type', 'url_length', 'number_of_dots_in_url', 'having_repeated_digits_in_url', 
                'number_of_digits_in_url', 'number_of_special_char_in_url', 'number_of_hyphens_in_url', 
                'number_of_underline_in_url', 'number_of_slash_in_url', 'number_of_questionmark_in_url', 
                'number_of_equal_in_url', 'number_of_at_in_url', 'number_of_dollar_in_url', 
                'number_of_exclamation_in_url', 'number_of_hashtag_in_url', 'number_of_percent_in_url', 
                'domain_length', 'number_of_dots_in_domain', 'number_of_hyphens_in_domain', 
                'having_special_characters_in_domain', 'number_of_special_characters_in_domain', 
                'having_digits_in_domain', 'number_of_digits_in_domain', 'having_repeated_digits_in_domain', 
                'number_of_subdomains', 'having_dot_in_subdomain', 'having_hyphen_in_subdomain', 
                'average_subdomain_length', 'average_number_of_dots_in_subdomain', 
                'average_number_of_hyphens_in_subdomain', 'having_special_characters_in_subdomain', 
                'number_of_special_characters_in_subdomain', 'having_digits_in_subdomain', 
                'number_of_digits_in_subdomain', 'having_repeated_digits_in_subdomain', 'having_path', 
                'path_length', 'having_query', 'having_fragment', 'having_anchor', 'entropy_of_url', 
                'entropy_of_domain']

    # Append the feedback and features to the CSV file
    with open('feedback.csv', 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writerow({'Type': feedback, **features})

    return jsonify(success=True)

if __name__ == "__main__":
    print("Starting the server and loading the model...")
    app.run(host='127.0.0.1', port=5000)
