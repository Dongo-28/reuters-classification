# 📊 Reuters News Classification with TensorFlow/Keras

Este projeto aplica **Deep Learning** para classificar notícias do conjunto de dados Reuters em diferentes tópicos. Utiliza redes neuronais densas (`Dense`) com a API `Sequential` do Keras.

---

## 📁 Estrutura do Projeto
```
reuters-classification/

├── .env

├── main.py

├── requirements.txt

└── README.md
```


---

## 🚀 Começar

### 🔽 Clonar o Repositório

```bash
git clone https://github.com/Dongo-28/reuters-classification.git
cd reuters-classification
```

### ⚙️ Instalar Dependências
```
Cria um ambiente virtual (opcional, mas recomendo):
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate   # macOS/Linux
```

Depois instala as dependências:
```
pip install -r requirements.txt

Exemplo de requirements.txt:
tensorflow==2.15.0
numpy
```
### ▶️ Executar o Projeto
python main.py
```
O script irá:
- Fazer o download automático do dataset Reuters.
- Pré-processar os dados.
- Construir e treinar o modelo.
- Avaliar no conjunto de teste.
```
### 📈 Resultados
```
Durante o treino, o modelo atinge uma acurácia de validação até 82%, e uma acurácia no conjunto de teste de aproximadamente 78%:
- Acurácia no conjunto de teste: 0.7836
```

### 🧠 Tecnologias Utilizadas
```
- Python 3.10+
- TensorFlow/Keras
- Reuters Dataset (via tf.keras.datasets)
- Rede Neuronal Feedforward (Fully Connected)
```

### 📌 Notas
```
- Avisos como Do not pass an input_shape to Dense são normais e não impedem a execução.
- Para melhor desempenho, considera usar EarlyStopping e Dropout.
```

### ✅ Tarefas Futuras
```
-  Adicionar visualização de gráficos de treino (matplotlib).
-  Melhorar generalização com Dropout.
-  Guardar o modelo treinado (model.save()).
```
---
### 🧑‍💻 Autor
```
Amadeu Dongo Rocha
```
---

### 📄 Licença
```
Este projeto é open-source e pode ser usado para fins educacionais. Licença: MIT.
```