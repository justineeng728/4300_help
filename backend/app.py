import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    fashion_df = pd.DataFrame(data['tshirts_and_tops'])

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(fashion_df['Description'])

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    query_vector = vectorizer.transform([query])

    cos_similarities = cosine_similarity(query_vector, tfidf_matrix)
    indices = cos_similarities.argsort()[0][::-1]
    
    top_matches_indices = indices[:5]
    top_matches = fashion_df.iloc[top_matches_indices][['Name', 'Price', 'Tagline', 'Description', 'ID', 'Image']]
    top_matches_json = top_matches.to_json(orient='records')
    
    return top_matches_json
    
    
@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/tshirts_and_tops")
def episodes_search():
    text = request.args.get("title")
    print("Query: " + str(text))
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
