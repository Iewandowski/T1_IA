from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Carregar o modelo MLP
alg_model = joblib.load('knn_model.pkl')

# Mapear classificação de estados do jogo
game_state_mapping = {
    'X': 'X Ganhou!',
    'O': 'O ganhou!',
    'DRAW': 'Empate',
    'NOT_FINISHED': 'Tem jogo'
}

def board_state_to_features(board_state):
    # Inicializa vetor com posições vazias
    features = np.zeros((27,), dtype=int)

    # Mapeia os valores do tabuleiro para índices no vetor de características
    symbol_mapping = {'X': 1, 'O': 2, 'b': 0}

    for i, symbol in enumerate(board_state):
        if symbol in symbol_mapping:
            features[i * 3 + symbol_mapping[symbol]] = 1

    return features

@app.route('/predict', methods=['POST'])
def predict():
    # Receber os dados de entrada do cliente
    data = request.json
    board_state = data['squares']  # Estado atual do jogo

    # Gerar as características apropriadas a partir do estado do tabuleiro
    features = board_state_to_features(board_state)

    # Fazer previsão
    prediction = alg_model.predict([features])

    # Interpretar a previsão para determinar o estado do jogo
    game_state = interpret_prediction(prediction)
    print("Dados recebidos do frontend:", data)
    return jsonify({'game_state': game_state})

def interpret_prediction(prediction):
    # Mapear a previsão para o estado do jogo usando o mapeamento definido acima
    return game_state_mapping[prediction[0]]

if __name__ == '__main__':
    app.run(debug=True)
