import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time

app = Flask(__name__)

# Inicializar o arquivo CSV se ele não existir
def initialize_data_file():
    try:
        df = pd.read_csv('dados.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['feature1','feature2','feature3','feature4','label'])
        df.to_csv('dados.csv', index=False)

# Salvar novos dados no CSV
def save_new_data(new_data):
    # Convertendo dados para DataFrame
    df = pd.DataFrame(new_data, columns=['feature1', 'feature2', 'feature3', 'feature4', 'label'])
    df.to_csv('dados_novos.csv', mode='a', header=False, index=False)

# Re-Treinar o modelo e salvar com um nome único
def retrain_model():
    if not os.path.exists('dados_novos.csv') or os.path.getsize('dados_novos.csv') == 0:
        return 'Nenhum dado disponivel para treinamento'

    # Carregar os dados armazenados
    df = pd.read_csv('dados_novos.csv')

    if df.shape[0] == 0:
        return 'Nenhum dado disponivel para treinamento'
    
    # Verifica se o numero de coluna é o esperado
    expected_columns = ['feature1', 'feature2', 'feature3', 'feature4', 'label']

    if list(df.columns) != expected_columns:
        return f'Formato de dados incorreto. Esperado: {expected_columns}'

    X = df.iloc[:, :-1].values # Features
    y = df.iloc[:, -1].values # Labels

    # Re-treinar o modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Salvar o modelo re-treinado com um nome único
    model_filename = f'modelo_random_forest{int(time.time())}.pkl'
    joblib.dump(model, model_filename)
    return f'Modelo re-treinado e salvo como {model_filename}'

# Rota para adicionar novos dados
@app.route('/add_data', methods=['POST'])
def app_data():
    data = request.get_json(force=True)
    save_new_data(data['input'])
    return jsonify({'status': 'Dados recebidos e armazenados'})

# Rota para re-treinar o modelo
@app.route('/retrain', methods=['POST'])
def retrein():
    model_filename = retrain_model()
    return jsonify({'status': f'Modelo re-treinado e salvo como {model_filename}'})

# Rota para fazer previsões
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    model = joblib.load('modelo_random_forest.pkl')

    # Espera-se que os dados sejam enviados como array JSON chamado 'input'
    excepted_num_features = len(model.feature_importances_) # Verifica se os numeros de caracteristicas corresponde ao esperado
    input_data = np.array(data['input']).reshape(1, -1)

    if input_data.shape[1] != excepted_num_features:
        return jsonify({'erro': f'O modelo espera {excepted_num_features} características, mas recebeu {input_data.shape[1]}'})

    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    initialize_data_file() # Inicializar o arquivo CSV
    app.run(port=5000, debug=True)