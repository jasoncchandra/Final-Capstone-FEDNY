from flask import Flask, request, jsonify
from models.sentiment_extractor import SentimentExtractor
from models.topic_gen import TopicGen
from models.vix_predictor import VixPredictor
from pyngrok import ngrok

app = Flask(__name__)

@app.route('/predict_vix', methods=['POST'])
def predict_vix():
    fomc_text = request.json.get('fomctext')
    if not fomc_text:
        return jsonify({'error': 'No FOMC text provided'}), 400
    
    # Dummy implementation for demonstration
    sentiment_extractor = SentimentExtractor()
    topic_gen = TopicGen()
    vix_predictor = VixPredictor()
    
    topic_vector = topic_gen.extract_topics(fomc_text)
    sentiment_score = sentiment_extractor.extract_sentiment(fomc_text)
    
    # Placeholder for previous VIX value
    previous_vix = 15.0
    
    predicted_vix = vix_predictor.predict(topic_vector, sentiment_score, previous_vix)
    return jsonify({'vix': predicted_vix})

@app.route('/get_vix_data', methods=['GET'])
def get_vix_data():
    latest_date = request.args.get('latestdate')
    # Dummy implementation for demonstration
    vix_data = dummy_get_vix_data(latest_date)
    return jsonify({'vix_data': vix_data})

def dummy_get_vix_data(latest_date):
    # Placeholder for the actual data retrieval logic
    return [{'date': '2023-01-01', 'vix': 15.0}, {'date': '2023-01-02', 'vix': 15.5}]

if __name__ == "__main__":
    public_url = ngrok.connect(5000)
    print(f"Public URL: {public_url}")
    app.run(host='0.0.0.0', port=5000)
