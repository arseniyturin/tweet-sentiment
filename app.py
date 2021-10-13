import os
#import csv
from flask import Flask, request, render_template
#import spacy
import pickle

with open('model_naivebayes', 'rb') as f:
    model = pickle.load(f)

os.environ['FLASK_ENVIRONMENT'] = 'development'
app = Flask(__name__)
#nlp = spacy.load('en_core_web_sm')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_sentiment', methods=['GET','POST'])
def get_sentiment():
    if request.method == 'POST':
        text = request.json['text']
        sentiment = model.predict([text])
        return sentiment[0]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
