# 🐶 Clasificador de Razas de Perros — Stanford Dogs Dataset

**Autor:** Juan Diego Chaparro García

---

## 📋 Descripción

Proyecto de clasificación de imágenes que entrena una red neuronal convolucional (CNN) para identificar **120 razas de perros** usando el [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). El pipeline incluye descarga del dataset, preprocesamiento con bounding boxes, data augmentation y entrenamiento con callbacks de regularización.

---

## 📁 Estructura del Proyecto

```
parcial.ipynb              # Notebook principal con todo el pipeline
/content/stanford_dogs/    # Dataset original (imágenes + anotaciones)
/content/stanford_dogs_processed/  # Imágenes recortadas y redimensionadas
model.keras                # Mejor modelo guardado automáticamente
```

---

## ⚙️ Requisitos

- Python 3.8+
- TensorFlow 2.x
- Pillow (PIL)
- NumPy
- Matplotlib

Instalación:
```bash
pip install tensorflow pillow numpy matplotlib
```

> Se recomienda ejecutar en **Google Colab** con GPU habilitada para mayor velocidad.

---

## 🚀 Pipeline

### 1. Verificación de GPU
Comprueba si hay GPU disponible. Si no, el entrenamiento corre en CPU (más lento).

### 2. Descarga del Dataset
Descarga automática desde Stanford Vision Lab:
- **Imágenes:** ~20,000 fotos de 120 razas
- **Anotaciones XML:** coordenadas del bounding box de cada perro

### 3. Preprocesamiento
- Lectura del bounding box desde los archivos XML de anotación
- Recorte de la región del perro en cada imagen
- Redimensionamiento a **150×150 píxeles**
- Validación de bounding boxes inválidos (evita imágenes negras)
- Imágenes guardadas por carpeta de raza en `/content/stanford_dogs_processed/`

### 4. Carga y Normalización
- División **90% entrenamiento / 10% validación** (seed=42)
- Batches de 32 imágenes
- Normalización de píxeles al rango `[0, 1]`

### 5. Data Augmentation
Aplicada solo en entrenamiento para mejorar la generalización:
- Flip horizontal aleatorio
- Rotación ±5%
- Zoom ±5%
- Contraste aleatorio
- Clip de valores para evitar artefactos

### 6. Arquitectura del Modelo (CNN)

```
Input: (150, 150, 3)
→ Data Augmentation
→ Conv2D(32) + LeakyReLU + MaxPooling
→ Conv2D(64) + LeakyReLU + MaxPooling
→ Conv2D(128) + LeakyReLU + MaxPooling
→ Conv2D(256) + LeakyReLU + MaxPooling
→ Flatten
→ Dense(32) + BatchNormalization
→ Dense(16) + BatchNormalization
→ Dense(120, softmax)
```

> **Nota:** Se usó `LeakyReLU` en lugar de `ReLU` para evitar el problema de "neuronas muertas" (dying ReLU) que generaba mapas de activación completamente negros.

### 7. Entrenamiento

- **Optimizador:** Adam (lr=0.0003)
- **Loss:** Sparse Categorical Crossentropy
- **Épocas máximas:** 60
- **Callbacks:**
  - `EarlyStopping` — detiene si `val_loss` no mejora en 5 épocas, restaura los mejores pesos
  - `ModelCheckpoint` — guarda automáticamente `model.keras` cuando mejora `val_loss`
  - `ReduceLROnPlateau` — reduce el learning rate ×0.3 si `val_loss` no mejora en 3 épocas (mínimo 1e-6)

### 8. Visualización y Evaluación
- Gráficas de accuracy y loss por época (entrenamiento vs validación)
- Accuracy de 36.20 en validation
- Identificación automática del mejor epoch según `val_loss`
- Feature maps de las 4 capas convolucionales para inspección visual
- Evaluación final en el conjunto de validación

---

## 📊 Ejecución desde Punto Intermedio

Si ya se tienen las imágenes preprocesadas en `/content/stanford_dogs_processed/`, se puede saltar directamente a la **celda 10** (`imagen_size = 150`) y ejecutar desde ahí.

---

## 📝 Notas Técnicas

- El tamaño de imagen `imagen_size = 150` está definido como variable global y es compartido por todas las etapas del pipeline.
- Las imágenes con bounding boxes inválidos (`xmin >= xmax` o `ymin >= ymax`) se omiten automáticamente con una advertencia.
- El dataset original contiene ~20,580 imágenes; el número final procesado puede variar ligeramente por imágenes corruptas o anotaciones faltantes.
