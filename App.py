# File: app.py

from flask import Flask, render_template_string, request, jsonify
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import pandas as pd

# --- Flask App ---
app = Flask(__name__)

model = None
vectorizer = None

# --- Preprocessing ---
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

# --- Load Model ---
def load_model():
    global model, vectorizer
    with open('trained_model.pkl', 'rb') as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
        vectorizer = saved_data['vectorizer']

# --- Predict Function ---
def predict_sentiment(text):
    processed_text = stemming(text)
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    prediction_proba = model.predict_proba(text_vector)[0]
    confidence = max(prediction_proba)

    sentiment = 'positive' if prediction == 1 else 'negative'
    percentage = int(confidence * 100)
    return {'sentiment': sentiment, 'confidence': confidence, 'percentage': percentage}

# --- Routes ---
@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Twitter Sentiment Analyzer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0d1117;
            --card-bg-color: rgba(22, 27, 34, 0.6);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-color: #e6edf3;
            --subtle-text-color: #8b949e;
            --accent-color-1: #58a6ff;
            --accent-color-2: #e88bff;
            --positive-color: #2da44e;
            --negative-color: #f85149;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Poppins', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            min-height: 100vh;
            padding: 2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
        }
        body::before, body::after {
            content: ''; position: absolute; z-index: -1;
            filter: blur(150px); opacity: 0.4;
        }
        body::before {
            width: 500px; height: 500px; border-radius: 50%;
            background: radial-gradient(circle, var(--accent-color-1), transparent 70%);
            top: -10%; left: -20%;
            animation: move-glow-1 15s ease-in-out infinite alternate;
        }
        body::after {
            width: 400px; height: 400px; border-radius: 50%;
            background: radial-gradient(circle, var(--accent-color-2), transparent 70%);
            bottom: -15%; right: -15%;
            animation: move-glow-2 12s ease-in-out infinite alternate;
        }
        @keyframes move-glow-1 { to { transform: translate(100px, 50px); } }
        @keyframes move-glow-2 { to { transform: translate(-80px, -40px); } }
        .container {
            background: var(--card-bg-color);
            backdrop-filter: blur(30px);
            border-radius: 24px; padding: 2.5rem 3rem;
            border: 1px solid var(--border-color); width: 100%;
            max-width: 650px; box-shadow: 0 16px 40px rgba(0,0,0,0.3);
        }
        .header { text-align: center; margin-bottom: 2.5rem; }
        .header h1 {
            font-size: 2.5em; font-weight: 700;
            background: linear-gradient(90deg, var(--accent-color-1), var(--accent-color-2));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .header p { font-size: 1.1em; color: var(--subtle-text-color); font-weight: 400; }
        .input-wrapper { position: relative; }
        textarea {
            width: 100%; min-height: 150px; padding: 1rem;
            background: rgba(13, 17, 23, 0.8); border: 1px solid var(--border-color);
            border-radius: 12px; font-size: 1rem; font-family: inherit;
            color: var(--text-color); resize: vertical; transition: all 0.3s ease; line-height: 1.6;
        }
        textarea:focus {
            outline: none; border-color: var(--accent-color-1);
            box-shadow: 0 0 15px rgba(88, 166, 255, 0.2);
        }
        textarea::placeholder { color: var(--subtle-text-color); }
        .analyze-button {
            display: flex; align-items: center; justify-content: center; gap: 0.75rem;
            width: 100%; background: linear-gradient(90deg, var(--accent-color-1), var(--accent-color-2));
            color: white; border: none; padding: 1rem; border-radius: 12px;
            font-size: 1.1rem; font-weight: 600; cursor: pointer;
            transition: all 0.3s ease; margin-top: 1.5rem; position: relative; overflow: hidden;
        }
        .analyze-button:hover:not(:disabled) {
            transform: translateY(-3px); box-shadow: 0 8px 25px rgba(88, 166, 255, 0.3);
        }
        .analyze-button .btn-text { transition: opacity 0.2s ease; }
        .analyze-button .spinner {
            width: 20px; height: 20px; border: 2px solid rgba(255,255,255,0.3);
            border-top-color: white; border-radius: 50%;
            animation: spin 0.8s linear infinite; position: absolute;
            opacity: 0; transition: opacity 0.2s ease;
        }
        .analyze-button.loading .spinner { opacity: 1; }
        .analyze-button.loading .btn-text { opacity: 0; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .result-section {
            margin-top: 2.5rem; min-height: 100px; display: flex;
            align-items: center; justify-content: center; opacity: 0;
            transform: translateY(20px); transition: opacity 0.5s ease, transform 0.5s ease;
        }
        .result-section.show { opacity: 1; transform: translateY(0); }
        
        /* === NEW --- Simplified Result Display Styles === */
        .result-content-simple {
            display: flex;
            align-items: center;
            gap: 1.5rem;
            width: 100%;
            background: rgba(255, 255, 255, 0.05);
            padding: 1.5rem;
            border-radius: 12px;
        }
        .sentiment-emoji {
            font-size: 3rem;
            animation: fadeIn 0.5s backwards;
        }
        .result-details-simple {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            animation: fadeIn 0.5s 0.2s backwards;
        }
        .result-label {
            font-size: 1.1rem;
            color: var(--subtle-text-color);
        }
        .sentiment-value {
            font-weight: 700;
            font-size: 1.2rem;
        }
        .sentiment-value.positive { color: var(--positive-color); }
        .sentiment-value.negative { color: var(--negative-color); }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 600px) {
            .container { padding: 2rem 1.5rem; }
            .header h1 { font-size: 2em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Twitter Sentiment Analyzer </h1>
        </div>
        <div class="input-wrapper">
            <textarea id="textInput" placeholder="Enter a sentence to analyze..."></textarea>
        </div>
        <button class="analyze-button" id="analyzeBtn" onclick="analyzeSentiment()">
            <span class="btn-text">Analyze Sentiment</span>
            <div class="spinner"></div>
        </button>
        <div class="result-section" id="resultSection"></div>
    </div>
    <script>
        const textInput = document.getElementById('textInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const resultSection = document.getElementById('resultSection');

        function showLoading(isLoading) {
            if (isLoading) {
                analyzeBtn.classList.add('loading');
                analyzeBtn.disabled = true;
            } else {
                analyzeBtn.classList.remove('loading');
                analyzeBtn.disabled = false;
            }
        }

        // --- JAVASCRIPT CHANGED ---
        // This function now builds the simple text display instead of the gauge
        function showResult(result) {
            const sentimentClass = result.sentiment.toLowerCase();
            const percentage = result.percentage;
            
            const emoji = sentimentClass === 'positive' ? '😊' : '😞';

            const resultHTML = `
                <div class="result-content-simple">
                    <div class="sentiment-emoji">${emoji}</div>
                    <div class="result-details-simple">
                        <p class="result-label">
                            Sentiment: <span class="sentiment-value ${sentimentClass}">${result.sentiment.toUpperCase()}</span>
                        </p>
                        <p class="result-label">
                            Confidence: <span class="sentiment-value">${percentage}%</span>
                        </p>
                    </div>
                </div>
            `;
            
            resultSection.innerHTML = resultHTML;
            resultSection.classList.add('show');
        }

        function showError(message) {
             resultSection.innerHTML = `<p style="color: var(--negative-color); text-align:center;">${message}</p>`;
             resultSection.classList.add('show');
        }

        async function analyzeSentiment() {
            const text = textInput.value.trim();
            if (!text) {
                showError('Please enter some text to analyze.');
                return;
            }

            showLoading(true);
            resultSection.classList.remove('show');
            await new Promise(resolve => setTimeout(resolve, 300));

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }
                const result = await response.json();
                
                if (result.error) {
                    showError(result.error);
                } else {
                    showResult(result);
                }
            } catch (error) {
                showError('Failed to connect to the server. Please try again.');
                console.error('Error:', error);
            } finally {
                showLoading(false);
            }
        }
        
        textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                analyzeBtn.click();
            }
        });
    </script>
</body>
</html>
    """)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json() or request.form
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'})
    return jsonify(predict_sentiment(text))

# --- Startup ---
if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
