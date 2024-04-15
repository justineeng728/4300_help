import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
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
aesthetics = os.path.join(current_directory, 'aesthetics.json')
merged = os.path.join(current_directory, 'merged_output.json')
# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    fashion_df = pd.DataFrame(data['tshirts_and_tops'])

with open(aesthetics,'r') as file:
    data2 = json.load(file)
    aesthetics_df = pd.DataFrame(data2['aesthetics'])

with open(merged,'r') as file:
    data3 =json.load(file)
    merged_df = pd.DataFrame(data3)

with open()

#all_fashion = pd.concat([fashion_df, high_knit_df])
split_index = len(all_fashion)
combined_df = pd.concat([merged, aesthetics_df])


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(combined_df['Description'])

svd = TruncatedSVD(n_components=3) 
tfidf_svd = svd.fit_transform(tfidf_matrix)

svd_fashion =tfidf_svd[:split_index]
svd_aesthetics = tfidf_svd[split_index:]

app = Flask(__name__)
CORS(app)

def svd_search(query):
    query_vector = vectorizer.transform([query])
    query_vector_svd = svd.transform(query_vector)

    cos_similarities = cosine_similarity(query_vector_svd, svd_fashion)
    print("Cosine Similarities:", cos_similarities)

    indices = cos_similarities.argsort()[0][::-1]
    print("Indices:", indices)

    top_matches_indices = indices[:5]
    top_matches = fashion_df.iloc[top_matches_indices][['Name', 'Price','Description','Image']]
    top_matches_json = top_matches.to_json(orient='records')

    return top_matches_json
    
    
@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/tshirts_and_tops")
def episodes_search():
    text = request.args.get("title")
    print("Query: " + str(text))
    return svd_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
