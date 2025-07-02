import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters

# 1. Carregar o dataset (somente as 10.000 palavras mais comuns)
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

# 2. Decodificar texto (opcional, para entender os dados)
# word_index = reuters.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])

# 3. Vetorização dos dados (binário)
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

x_train = vectorize_sequences(x_train)
x_test = vectorize_sequences(x_test)

# 4. Codificar as labels (one-hot)
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.0
    return results

y_train = to_one_hot(y_train)
y_test = to_one_hot(y_test)

# 5. Construção da rede
model = keras.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))  # 46 classes

# 6. Compilação
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 7. Divisão de validação
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# 8. Treinar o modelo
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# 9. Avaliar no conjunto de teste
results = model.evaluate(x_test, y_test)
print(f"Acurácia no conjunto de teste: {results[1]:.4f}")