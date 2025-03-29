import json
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

def train_model():
    with open("intents.json", "r", encoding="utf-8") as file:
        intents = json.load(file)
    
    print("Carregando intenções:", intents["intents"])

    words = []
    classes = []
    documents = []
    ignore_words = ["?", "!", ".", ","]

    # Processar as intenções
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    print("Palavras:", words)
    print("Classes:", classes)

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(set(words))
    classes = sorted(set(classes))

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

    print("Treinando com:", len(training), "amostras")

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = np.array([x[0] for x in training])
    train_y = np.array([x[1] for x in training])

    # Criar o modelo de rede neural
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
    print("✅ Modelo treinado e salvo com sucesso!")
