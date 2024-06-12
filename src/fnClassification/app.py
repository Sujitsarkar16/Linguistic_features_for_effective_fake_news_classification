from flask import Flask, render_template, request
import pickle
from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
import numpy as np
from textstat.textstat import textstatistics

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load the model
loaded_model = pickle.load(
    open("./src/fnClassification/Models/XG_boost.pkl", 'rb'))

lemmatizer = WordNetLemmatizer()
stopwords_1 = set(stopwords.words('english'))


def extract_count_features(text):
    tokens = word_tokenize(text)
    sentences = sent_tokenize(text)
    tagged = pos_tag(tokens)

    pos_counts = nltk.FreqDist(tag for _, tag in tagged)
    readability = textstatistics()

    features = {
        'Pronouns': sum(1 for _, tag in tagged if tag in ['PRP', 'PRP$']),
        'TO': pos_counts.get('TO', 0),
        'Key_conectors': sum(1 for word in tokens if word.lower() in ['and', 'but', 'or', 'so']),
        'Flesch_Kincaid_Grade_Level': readability.flesch_kincaid_grade(text),
        'Flesch_Reading_Ease': readability.flesch_reading_ease(text),
        'CLI': readability.coleman_liau_index(text),
        'add_info': 0,
        'Linsear_write_formula': readability.linsear_write_formula(text),
        'Determiners': pos_counts.get('DT', 0),
        'ARI': readability.automated_readability_index(text),
        'Number_of_Words': len(tokens),
        'LIWC_pronouns': 0,
        'Negations': sum(1 for word in tokens if word.lower() in ['not', 'no', 'never', 'none']),
        'NNP': pos_counts.get('NNP', 0),
        'TPP': 0,
        'PRP': pos_counts.get('PRP', 0),
        'Positive_Words': 0,
        'Coleman_Liau_Index': readability.coleman_liau_index(text),
        'DT': pos_counts.get('DT', 0),
        'RB': pos_counts.get('RB', 0),
        'Number_of_Words_per_Sentence': np.mean([len(word_tokenize(sentence)) for sentence in sentences]),
        'CC': pos_counts.get('CC', 0),
        'Number_of_Types': len(set(tokens))
    }
    return features


count_features = ['Pronouns', 'TO', 'Key_conectors', 'Flesch_Kincaid_Grade_Level',
                  'Flesch_Reading_Ease', 'CLI', 'add_info', 'Linsear_write_formula',
                  'Determiners', 'ARI', 'Number_of_Words', 'LIWC_pronouns', 'Negations',
                  'NNP', 'TPP', 'PRP', 'Positive_Words', 'Coleman_Liau_Index', 'DT', 'RB',
                  'Number_of_Words_per_Sentence', 'CC', 'Number_of_Types']


def fake_news_prediction(news):
    news = re.sub(r'[^a-zA-Z\s]', '', news)
    features_count = extract_count_features(news)

    feature_array = np.zeros(len(count_features))
    for i, feature in enumerate(count_features):
        feature_array[i] = features_count.get(feature, 0)

    # Reshape to match model input
    feature_array = feature_array.reshape(1, -1)

    prediction = loaded_model.predict(feature_array)
    return prediction


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        if not message:
            return render_template("predict.html", prediction="Please enter some news text to predict.")
        pred = fake_news_prediction(message)
        result = "Prediction news: It is Real News📰" if pred[
            0] == 1 else "Prediction news: It is Fake News"
        return render_template("predict.html", prediction_text=result)
    else:
        return render_template('predict.html', prediction="Something went wrong")


if __name__ == '__main__':
    app.run(debug=True)
