# DocumentacionINF-740

# **Documentación del Código: Análisis de Sentimientos con Redes Neuronales Recurrentes**

## **Objetivo del Código**
Este código implementa un modelo de red neuronal recurrente para realizar análisis de sentimientos en comentarios de películas, utilizando el conjunto de datos **IMDB** proporcionado por Keras. El objetivo es clasificar los comentarios como **positivos** o **negativos** en función de su contenido textual.

---

## **Descripción del Proyecto**
### 1. **Datos**
El modelo utiliza el dataset **IMDB**, una base de datos de comentarios de películas ampliamente utilizada para tareas de procesamiento de lenguaje natural (NLP). Este dataset contiene:
- **Comentarios preprocesados**: Representados como secuencias de índices de palabras.
- **Etiquetas binarias**:
  - `1`: Indica un comentario positivo.
  - `0`: Indica un comentario negativo.

### 2. **Modelo**
Se implementa un modelo basado en **Redes Neuronales Recurrentes (RNN)** con celdas **LSTM (Long Short-Term Memory)**, conocidas por su capacidad de trabajar con datos secuenciales como texto. 

---

## **Explicación del Código**

### **1. Importación de librerías**
```python
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
```
Se importan las librerías necesarias:
- **NumPy**: Para trabajar con arreglos y datos numéricos.
- **Keras**: Para implementar el modelo y procesar los datos textuales.
  - `pad_sequences`: Normaliza las longitudes de las secuencias de texto.
  - `Tokenizer`: No se usa en este código, pero es útil para tokenizar texto.
  - Módulo de redes neuronales (`Sequential`, capas `Embedding`, `LSTM`, etc.).

---

### **2. Carga de datos**
```python
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
```
Se cargan los datos de entrenamiento y prueba del dataset IMDB:
- **`X_train` y `X_test`**: Comentarios representados como secuencias de números (tokens).
- **`y_train` y `y_test`**: Etiquetas binarias (1 = positivo, 0 = negativo).
- **`num_words=10000`**: Se limitan a las 10,000 palabras más comunes para reducir la complejidad.

---

### **3. Preprocesamiento de las secuencias**
```python
X_train = pad_sequences(X_train, maxlen=150)
X_test = pad_sequences(X_test, maxlen=150)
```
Se normalizan las secuencias para que todas tengan la misma longitud:
- **`maxlen=150`**: Cada comentario se ajusta a 150 palabras. Si es más corto, se rellena con ceros al inicio.

---

### **4. Visualización de un comentario y su etiqueta**
```python
palabra_a_id = keras.datasets.imdb.get_word_index()
id_a_palabra = {i: palabra for palabra, i in palabra_a_id.items()}
print('Comentario:' + str([id_a_palabra.get(i,'') for i in X_train[6]]))
print('Etiqueta:' + str(y_train[6]))
```
- **`get_word_index()`**: Devuelve un diccionario que asocia palabras con índices.
- **`id_a_palabra`**: Convierte los índices de palabras nuevamente a texto legible.
- **Salida**: Muestra un comentario de entrenamiento (secuencia convertida a texto) y su etiqueta.

---

### **5. Construcción del modelo**
```python
model = keras.models.Sequential([
    keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=150),
    keras.layers.LSTM(32, dropout=0.1, recurrent_dropout=0.2, return_sequences=True),
    keras.layers.LSTM(32, dropout=0.1, recurrent_dropout=0.2),
    keras.layers.Dense(1, activation='sigmoid')
])
```
El modelo se compone de las siguientes capas:
1. **Embedding Layer**:
   - **`input_dim=10000`**: El vocabulario tiene 10,000 palabras.
   - **`output_dim=32`**: Cada palabra se representa como un vector de 32 dimensiones.
   - **`input_length=150`**: Se espera una longitud fija de 150 palabras.

2. **Capa LSTM** (Primera):
   - **`32` unidades**: Define la dimensión del estado oculto.
   - **`dropout=0.1`**: Reduce el sobreajuste descartando un 10% de las conexiones.
   - **`recurrent_dropout=0.2`**: Descartes aplicados a los estados internos.

3. **Capa LSTM** (Segunda):
   - Similar a la primera, pero sin devolver secuencias intermedias (`return_sequences=False`).

4. **Capa Densa (Dense)**:
   - **`Dense(1)`**: Produce una salida escalar (probabilidad).
   - **`activation='sigmoid'`**: Mapea la salida entre 0 (negativo) y 1 (positivo).

---

### **6. Compilación y entrenamiento**
```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1, validation_data=(X_test, y_test))
```
- **`loss='binary_crossentropy'`**: Función de pérdida para problemas de clasificación binaria.
- **`optimizer='adam'`**: Algoritmo de optimización avanzado.
- **`epochs=5`**: Entrena el modelo durante 5 iteraciones completas sobre los datos.
- **`batch_size=64`**: Procesa 64 ejemplos por vez.

---

### **7. Evaluación del modelo**
```python
scores = model.evaluate(X_test, y_test, verbose=0)
print('Exactitud:{:.2f}'.format(scores[1]))
```
- Se evalúa el modelo con los datos de prueba.
- **`scores[1]`**: Representa la métrica de exactitud (accuracy).

---

### **8. Predicción sobre nuevos ejemplos**
```python
texto_neg = X_test[9]
texto_pos = X_test[13]
texts = (texto_neg, texto_pos)
textos = pad_sequences(texts, maxlen=300, value=0.0)
preds = model.predict(textos)
print("predicciones:", preds)
```
- Se seleccionan dos comentarios del conjunto de prueba.
- Se ajustan sus longitudes a 300 palabras con `pad_sequences`.
- **`model.predict`**: Genera predicciones de probabilidad para cada comentario:
  - Valores cercanos a 0 → Sentimiento negativo.
  - Valores cercanos a 1 → Sentimiento positivo.

---

## **Resultados Esperados**
1. **Exactitud**: Muestra la capacidad del modelo para clasificar correctamente comentarios en el conjunto de prueba.
2. **Predicciones**: Para nuevos comentarios, el modelo indica si son positivos o negativos.

---

## **Conclusión**
Este código demuestra el uso de redes neuronales recurrentes para tareas de análisis de sentimientos. Implementa técnicas modernas como embeddings y LSTM para capturar relaciones contextuales en el texto. Es un ejemplo práctico de cómo las redes neuronales pueden abordar problemas reales de procesamiento de lenguaje natural (NLP).
