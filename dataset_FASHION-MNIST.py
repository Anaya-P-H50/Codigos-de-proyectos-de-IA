# ============================================================
# RECONOCIMIENTO DE PRENDAS DE VESTIR - FASHION-MNIST
# Machine Learning: Aprendizaje supervisado
# Red neuronal: CNN (Convolutional Neural Network)
# Framework: TensorFlow / Keras
# ============================================================

import os
import numpy as np
import tensorflow as tf
import tkinter as tk

from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps


# ============================================================
# 1. CONFIGURACION GENERAL
# ============================================================

CATEGORIAS = [
    "Camiseta (Top)",
    "Pantalon",
    "Jersey (Sueter)",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla deportiva",
    "Bolso",
    "Botin"
]

RUTA_MODELO = "fashion_mnist_cnn.keras"

EPOCAS = 5
TAMANO_LOTE = 128


# ============================================================
# 2. CARGA DEL DATASET FASHION-MNIST
# ============================================================

print("\nCargando dataset Fashion-MNIST...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

print("Imagenes de entrenamiento:", len(x_train))
print("Imagenes de prueba:", len(x_test))


# ============================================================
# 3. PREPROCESAMIENTO DE LAS IMAGENES
# ============================================================

# Normalizar los valores de los pixeles de 0-255 a 0-1.

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Agregar la dimension del canal de color (escala de grises).

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("\nDimensiones de las imagenes:", x_train.shape)


# ============================================================
# 4. CONSTRUCCION DE LA RED NEURONAL CONVOLUCIONAL (CNN)
# ============================================================

def crear_modelo():
    modelo = tf.keras.Sequential([
        # Primera capa convolucional
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(28, 28, 1)
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Segunda capa convolucional
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Convertir caracteristicas en un vector
        tf.keras.layers.Flatten(),
        # Capa neuronal completamente conectada
        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),
        # Reducir el sobreajuste
        tf.keras.layers.Dropout(0.3),
        # Capa de salida: 10 categorias
        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )
    ])
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return modelo


# ============================================================
# 5. ENTRENAMIENTO DEL MODELO
# ============================================================

if os.path.exists(RUTA_MODELO):

    print("\nSe encontro un modelo previamente entrenado.")

    modelo = tf.keras.models.load_model(RUTA_MODELO)

else:

    print("\nConstruyendo red neuronal...")

    modelo = crear_modelo()

    print("\nIniciando entrenamiento de Machine Learning...")

    modelo.fit(

        x_train,
        y_train,

        epochs=EPOCAS,

        batch_size=TAMANO_LOTE,

        validation_split=0.1,

        verbose=1
    )

    # Guardar el modelo entrenado
    modelo.save(RUTA_MODELO)

    print("\nModelo guardado correctamente.")


# ============================================================
# 6. EVALUACION DEL MODELO
# ============================================================

print("\nEvaluando el modelo con las imagenes de prueba...")

perdida, precision = modelo.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\nPrecision obtenida:",
      round(precision * 100, 2), "%")


# ============================================================
# 7. FUNCION DE CLASIFICACION
# ============================================================

def clasificar_imagen(imagen):

    # Preparar la imagen
    imagen = imagen.astype("float32") / 255.0

    imagen = np.expand_dims(imagen, axis=-1)

    imagen = np.expand_dims(imagen, axis=0)

    # Realizar prediccion
    predicciones = modelo.predict(
        imagen,
        verbose=0
    )[0]

    # Obtener las tres categorias mas probables
    mejores = np.argsort(predicciones)[-3:][::-1]

    return predicciones, mejores


# ============================================================
# 8. INTERFAZ GRAFICA CON TKINTER
# ============================================================

ventana = tk.Tk()

ventana.title("Fashion-MNIST - Reconocimiento de ropa")

ventana.geometry("650x650")

ventana.configure(bg="#202936")


# Titulo principal

titulo = tk.Label(

    ventana,

    text="CLASIFICADOR INTELIGENTE DE ROPA",

    font=("Arial", 17, "bold"),

    bg="#202936",

    fg="white"

)

titulo.pack(pady=20)


# Informacion del proyecto

descripcion = tk.Label(

    ventana,

    text="Red neuronal convolucional CNN | 10 categorias",

    font=("Arial", 10),

    bg="#202936",

    fg="#BFC9D5"

)

descripcion.pack(pady=5)


# Espacio para mostrar la imagen

etiqueta_imagen = tk.Label(

    ventana,

    text="Selecciona una imagen para comenzar",

    width=30,

    height=12,

    bg="#FFFFFF",

    fg="black"

)

etiqueta_imagen.pack(pady=20)


# Resultado de la clasificacion

resultado = tk.Label(

    ventana,

    text="Resultado: esperando imagen...",

    font=("Arial", 14, "bold"),

    bg="#202936",

    fg="#62D9A5"

)

resultado.pack(pady=10)


# Resultados secundarios

detalle = tk.Label(

    ventana,

    text="",

    font=("Arial", 11),

    bg="#202936",

    fg="white",

    justify="left"

)

detalle.pack(pady=10)


# Variable para conservar la imagen mostrada

imagen_mostrada = None


# ============================================================
# 9. FUNCION PARA MOSTRAR Y CLASIFICAR UNA IMAGEN
# ============================================================

def mostrar_resultado(imagen):
    global imagen_mostrada
    predicciones, mejores = clasificar_imagen(imagen)
    # Mostrar la imagen ampliada
    imagen_pil = Image.fromarray(
        (imagen * 255).astype("uint8")
    )
    imagen_pil = imagen_pil.resize(
        (224, 224),
        Image.Resampling.NEAREST
    )
    imagen_mostrada = ImageTk.PhotoImage(imagen_pil)
    etiqueta_imagen.configure(
        image=imagen_mostrada,
        text=""
    )
    # Categoria principal
    indice = mejores[0]
    confianza = predicciones[indice] * 100
    resultado.configure(
        text=(
            "Prenda identificada: "
            + CATEGORIAS[indice]
            + f"\nProbabilidad: {confianza:.2f}%"
        )
    )
    # Mostrar las tres predicciones principales
    texto = "Otras posibilidades:\n\n"
    for posicion in mejores:
        porcentaje = predicciones[posicion] * 100
        texto += (
            CATEGORIAS[posicion]
            + f" : {porcentaje:.2f}%\n"
        )
    detalle.configure(text=texto)


# ============================================================
# 10. SELECCIONAR UNA IMAGEN DEL DATASET
# ============================================================

def probar_dataset():

    indice = np.random.randint(0, len(x_test))

    imagen = x_test[indice].squeeze()

    mostrar_resultado(imagen)


# ============================================================
# 11. CARGAR UNA IMAGEN EXTERNA
# ============================================================

def cargar_imagen():

    ruta = filedialog.askopenfilename(

        title="Seleccionar imagen de ropa",

        filetypes=[
            ("Imagenes", "*.png *.jpg *.jpeg *.bmp")
        ]

    )

    if not ruta:
        return

    try:

        imagen = Image.open(ruta).convert("L")

        # Ajustar la imagen a 28x28 pixeles
        imagen = ImageOps.fit(
            imagen,
            (28, 28)
        )

        imagen = np.array(imagen)

        mostrar_resultado(imagen)

    except Exception as error:

        messagebox.showerror(
            "Error",
            str(error)
        )


# ============================================================
# 12. BOTONES DE LA INTERFAZ
# ============================================================

boton_dataset = tk.Button(

    ventana,

    text="Probar imagen del dataset",

    command=probar_dataset,

    font=("Arial", 11, "bold"),

    bg="#2878B5",

    fg="white",

    width=28,

    height=2

)

boton_dataset.pack(pady=8)


boton_cargar = tk.Button(

    ventana,

    text="Cargar imagen desde mi computadora",

    command=cargar_imagen,

    font=("Arial", 11, "bold"),

    bg="#269B70",

    fg="white",

    width=32,

    height=2

)

boton_cargar.pack(pady=8)


# ============================================================
# 13. INICIAR INTERFAZ
# ============================================================

print("\nInterfaz grafica iniciada.")

ventana.mainloop()