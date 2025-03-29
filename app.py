import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from flask import Flask, render_template, request, jsonify

# Baixar os recursos necessários do NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

# Carregar os dados de treinamento
try:
    words = pickle.load(open("model/words.pkl", "rb"))
    classes = pickle.load(open("model/classes.pkl", "rb"))
except FileNotFoundError:
    words = []
    classes = []

# Tentar carregar o modelo treinado (se existir)
try:
    model = load_model("model/chatbot_model.h5")
except Exception as e:
    model = None
    print(f"Erro ao carregar modelo: {e}")

# Carregar o arquivo de intenções
try:
    with open("intents.json", "r", encoding="utf-8") as file:
        intents = json.load(file)
except Exception as e:
    print(f"Erro ao carregar intents.json: {e}")
    intents = {"intents": []}  # Em caso de erro, criar uma lista vazia de intenções

# Função para processar a entrada do usuário
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_intent(sentence):
    if model is None:
        print("Modelo não encontrado! Iniciando o treinamento...")
        train_model()

    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Função para treinar o modelo
def train_model():
    global words, classes  # Declarando que 'words' e 'classes' são variáveis globais
    # Criar os dados de treinamento
    documents = []
    ignore_words = ["?", "!", ".", ","]
    
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(set(words))
    classes = sorted(set(classes))

    pickle.dump(words, open("model/words.pkl", "wb"))
    pickle.dump(classes, open("model/classes.pkl", "wb"))

    # Criar os dados de treinamento
    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        word_patterns = doc[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = np.array([x[0] for x in training])
    train_y = np.array([x[1] for x in training])

    # Criar o modelo de rede neural
    global model  # Garantir que o modelo seja acessado globalmente
    model = Sequential([
        Dense(128, input_shape=(len(train_x[0]),), activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(len(train_y[0]), activation="softmax")
    ])

    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

    # Treinar o modelo
    model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

    # Salvar o modelo treinado
    model.save("model/chatbot_model.h5")
    print("Modelo treinado e salvo com sucesso!")

# Função para adicionar novas intenções
def save_new_intent(pattern, tag, response):
    new_intent = {
        "tag": tag,
        "patterns": [pattern],
        "responses": [response]
    }

    intents["intents"].append(new_intent)
    with open("intents.json", "w", encoding="utf-8") as file:
        json.dump(intents, file, indent=4)

    train_model()

# Rota Flask para a interface
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")  # Usando `.get()` para evitar erro caso a chave não exista

    if not user_message:
        return jsonify({"response": "Desculpe, não entendi."})

    # Verificar a intenção com o modelo treinado
    intents_prediction = predict_intent(user_message)
    
    if intents_prediction:
        tag = intents_prediction[0]["intent"]
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                response = random.choice(intent["responses"])
                return jsonify({"response": response})
    
    return jsonify({"response": "Desculpe, não entendi."})

@app.route("/learn", methods=["POST"])
def learn():
    user_message = request.json.get("message")
    user_tag = request.json.get("tag")
    user_response = request.json.get("response")

    if not all([user_message, user_tag, user_response]):
        return jsonify({"response": "Todos os campos são necessários para o aprendizado."})

    # Adiciona a nova intenção e treina o modelo
    save_new_intent(user_message, user_tag, user_response)

    return jsonify({"response": "Obrigado por me ensinar! Estou aprendendo."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
