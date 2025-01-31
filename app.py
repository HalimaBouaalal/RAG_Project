from flask import Flask, request, jsonify, render_template
import requests
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

# Configure embedding model
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434"
)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    # Récupérer les données JSON envoyées par la requête
    data = request.get_json()
    print("Données reçues :", data)

    # Vérifier si la clé 'question' existe dans les données
    question = data['instances'][0]['question']
    print(question)
    if not question:
        return jsonify({"error": "La question n'a pas été fournie."}), 400
    url = "http://127.0.0.1:5001/invocations"
    

    try:
        response = requests.post(url, json=data)
        response_data = response.json()
        print(response_data)
        return jsonify({"answer": response_data.get('predictions', 'No answer received')})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    # Lancer le serveur Flask sur le port 8000
    app.run(host="127.0.0.1", port=8000, debug=True)
