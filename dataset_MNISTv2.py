# RECONOCIMIENTO DE DIGITOS ESCRITOS A MANO - MNIST
# Machine Learning: Aprendizaje supervisado
# Red neuronal convolucional (CNN)
# TensorFlow + PyQt6 + Pillow + NumPy

import sys
import numpy as np

from pathlib import Path

import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

from PIL import Image, ImageOps

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox
)

from PyQt6.QtCore import Qt, QThread, pyqtSignal

from PyQt6.QtGui import (
    QImage,
    QPainter,
    QPen,
    QColor,
    QFont
)


# ============================================================
# 1. CONFIGURACION GENERAL
# ============================================================

RUTA_MODELO = Path(__file__).with_name("modelo_mnist.keras")


# ============================================================
# 2. CREACION DE LA RED NEURONAL CNN
# ============================================================

def crear_modelo():

    red = models.Sequential([

        # Entrada de imagen de 28x28 pixeles
        layers.Input(shape=(28, 28, 1)),

        # Primera capa convolucional
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D((2, 2)),

        # Segunda capa convolucional
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D((2, 2)),

        # Convertir caracteristicas en vector
        layers.Flatten(),

        # Capa neuronal
        layers.Dense(
            64,
            activation="relu"
        ),

        # Clasificar los diez digitos
        layers.Dense(
            10,
            activation="softmax"
        )

    ])

    red.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return red


# ============================================================
# 3. ENTRENAMIENTO DEL MODELO
# ============================================================

class Entrenamiento(QThread):

    estado = pyqtSignal(str)

    terminado = pyqtSignal(object, float)

    error = pyqtSignal(str)

    def run(self):

        try:

            # Comprobar si existe un modelo guardado
            if RUTA_MODELO.exists():

                self.estado.emit(
                    "Cargando modelo entrenado..."
                )

                red = tf.keras.models.load_model(
                    RUTA_MODELO
                )

                # Comprobar que tenga diez salidas
                if red.output_shape[-1] != 10:
                    raise ValueError(
                        "El modelo guardado no tiene "
                        "las diez categorias esperadas."
                    )

                self.terminado.emit(red, -1.0)

                return

            # Descargar dataset
            self.estado.emit(
                "Descargando dataset MNIST..."
            )

            # Cargar el dataset MNIST
            #Aprendizaje 
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            # Utilizar una cantidad menor de imágenes para comenzar
            x_train = x_train[:10000]
            y_train = y_train[:10000]

            # Normalizar los valores de las imágenes
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0

            # Adaptar las imágenes a la red neuronal
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)

            # Crear la red neuronal
            red = crear_modelo()

            # Entrenar el modelo
            red.fit(
                x_train,
                y_train,
                epochs=3,
                batch_size=256,
                validation_split=0.1,
                verbose=1
            )

            # Evaluar modelo
            self.estado.emit(
                "Evaluando el modelo..."
            )

            perdida, precision = red.evaluate(
                x_test,
                y_test,
                verbose=0
            )

            # Guardar modelo
            red.save(RUTA_MODELO)

            self.terminado.emit(
                red,
                float(precision * 100)
            )

        except Exception as e:

            self.error.emit(str(e))


# ============================================================
# 4. PIZARRA DIGITAL
# ============================================================

class Pizarra(QWidget):

    def __init__(self):

        super().__init__()

        self.setFixedSize(320, 320)

        # Imagen blanca donde se dibujara
        self.imagen = QImage(
            320,
            320,
            QImage.Format.Format_RGB32
        )

        self.imagen.fill(QColor("white"))

        self.dibujando = False

        self.ultimo_punto = None

        self.setCursor(
            Qt.CursorShape.CrossCursor
        )

    # Mostrar imagen en pantalla
    def paintEvent(self, event):

        pintor = QPainter(self)

        pintor.drawImage(
            0,
            0,
            self.imagen
        )

        pintor.end()

    # Detectar clic del mouse
    def mousePressEvent(self, event):

        if event.button() == Qt.MouseButton.LeftButton:

            self.dibujando = True

            self.ultimo_punto = (
                event.position().toPoint()
            )

            # Dibujar punto inicial
            pintor = QPainter(self.imagen)

            lapiz = QPen(QColor("black"))

            lapiz.setWidth(18)

            lapiz.setCapStyle(
                Qt.PenCapStyle.RoundCap
            )

            pintor.setPen(lapiz)

            pintor.drawPoint(
                self.ultimo_punto
            )

            pintor.end()

            self.update()

    # Detectar movimiento del mouse
    def mouseMoveEvent(self, event):

        if self.dibujando:

            punto = event.position().toPoint()

            pintor = QPainter(self.imagen)

            lapiz = QPen(QColor("black"))

            lapiz.setWidth(18)

            lapiz.setCapStyle(
                Qt.PenCapStyle.RoundCap
            )

            lapiz.setJoinStyle(
                Qt.PenJoinStyle.RoundJoin
            )

            pintor.setPen(lapiz)

            pintor.drawLine(
                self.ultimo_punto,
                punto
            )

            pintor.end()

            self.ultimo_punto = punto

            self.update()

    # Terminar dibujo
    def mouseReleaseEvent(self, event):

        if event.button() == Qt.MouseButton.LeftButton:

            self.dibujando = False

            self.ultimo_punto = None

    # Borrar pizarra
    def borrar(self):

        self.imagen.fill(
            QColor("white")
        )

        self.update()

    # Obtener imagen para TensorFlow
    def obtener_imagen(self):

        # Convertir QImage a escala de grises
        imagen_gris = self.imagen.convertToFormat(
            QImage.Format.Format_Grayscale8
        )

        ancho = imagen_gris.width()

        alto = imagen_gris.height()

        bytes_por_linea = imagen_gris.bytesPerLine()

        # Obtener los datos de los pixeles
        bits = imagen_gris.bits()

        bits.setsize(
            imagen_gris.sizeInBytes()
        )

        arreglo = np.frombuffer(
            bits,
            dtype=np.uint8
        )

        # Considerar el relleno de cada linea
        arreglo = arreglo.reshape(
            alto,
            bytes_por_linea
        )

        # Eliminar bytes adicionales
        arreglo = arreglo[:, :ancho]

        # Crear imagen PIL independiente
        imagen_pil = Image.fromarray(
            arreglo.copy()
        )

        return imagen_pil


# ============================================================
# 5. PROCESAMIENTO DE LA IMAGEN
# ============================================================

def preparar_imagen(imagen):

    # Invertir colores
    # Fondo negro y numero blanco
    procesada = ImageOps.invert(imagen)

    # Encontrar limites del numero
    mascara = procesada.point(
        lambda p: 255 if p > 30 else 0
    )

    limites = mascara.getbbox()

    # Comprobar si hay dibujo
    if limites is None:

        return None

    # Recortar numero
    numero = procesada.crop(limites)

    # Ajustar a 20x20 manteniendo proporcion
    numero.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Crear imagen de 28x28
    resultado = Image.new(
        "L",
        (28, 28),
        0
    )

    # Centrar numero
    x = (28 - numero.width) // 2

    y = (28 - numero.height) // 2

    resultado.paste(
        numero,
        (x, y)
    )

    # Normalizar imagen
    arreglo = np.array(
        resultado
    ).astype("float32") / 255.0

    # Adaptar a la entrada de CNN
    arreglo = arreglo.reshape(
        1,
        28,
        28,
        1
    )

    return arreglo


# ============================================================
# 6. INTERFAZ PRINCIPAL
# ============================================================

class Aplicacion(QMainWindow):

    def __init__(self):

        super().__init__()

        self.modelo = None

        self.setWindowTitle(
            "Reconocimiento de Digitos - MNIST"
        )

        self.setFixedSize(480, 690)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #F1F5F9;
            }

            QLabel {
                color: #1E293B;
            }

            QPushButton {
                color: white;
                border: none;
                border-radius: 7px;
                padding: 12px;
                font-size: 12px;
                font-weight: bold;
            }

            QPushButton:disabled {
                background-color: #94A3B8;
            }
        """)

        # Contenedor principal
        central = QWidget()

        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        layout.setAlignment(
            Qt.AlignmentFlag.AlignHCenter
        )

        layout.setSpacing(12)

        
        # TITULO
        
        titulo = QLabel(
            "RECONOCIMIENTO DE DIGITOS"
        )

        titulo.setFont(
            QFont(
                "Arial",
                17,
                QFont.Weight.Bold
            )
        )

        titulo.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        layout.addWidget(titulo)

        # ----------------------------------------------------
        # INSTRUCCIONES
        # ----------------------------------------------------

        instruccion = QLabel(
            "Por favor, dibuja un numero del 0 al 9"
        )

        instruccion.setFont(
            QFont("Arial", 11)
        )

        instruccion.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        layout.addWidget(instruccion)

        # ----------------------------------------------------
        # PIZARRA
        # ----------------------------------------------------

        self.pizarra = Pizarra()

        self.pizarra.setStyleSheet("""
            border: 2px solid #CBD5E1;
            background-color: white;
        """)

        layout.addWidget(
            self.pizarra,
            alignment=Qt.AlignmentFlag.AlignCenter
        )

        # ----------------------------------------------------
        # BOTONES
        # ----------------------------------------------------

        botones = QHBoxLayout()

        self.btn_reconocer = QPushButton(
            "Reconocer numero"
        )

        self.btn_reconocer.setStyleSheet(
            "background-color: #2563EB;"
        )

        self.btn_reconocer.clicked.connect(
            self.reconocer
        )

        # Desactivar hasta cargar el modelo
        self.btn_reconocer.setEnabled(False)

        self.btn_borrar = QPushButton(
            "Borrar dibujo"
        )

        self.btn_borrar.setStyleSheet(
            "background-color: #64748B;"
        )

        self.btn_borrar.clicked.connect(
            self.borrar
        )

        botones.addWidget(
            self.btn_reconocer
        )

        botones.addWidget(
            self.btn_borrar
        )

        layout.addLayout(botones)

        # ----------------------------------------------------
        # RESULTADO
        # ----------------------------------------------------

        self.resultado = QLabel(
            "Numero reconocido: --"
        )

        self.resultado.setFont(
            QFont(
                "Arial",
                17,
                QFont.Weight.Bold
            )
        )

        self.resultado.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        layout.addWidget(
            self.resultado
        )

        # ----------------------------------------------------
        # CONFIANZA
        # ----------------------------------------------------

        self.confianza = QLabel(
            "Confianza del modelo: --"
        )

        self.confianza.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        layout.addWidget(
            self.confianza
        )

        # ----------------------------------------------------
        # OTRAS POSIBILIDADES
        # ----------------------------------------------------

        self.posibilidades = QLabel(
            "Esperando un dibujo..."
        )

        self.posibilidades.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        layout.addWidget(
            self.posibilidades
        )

        # ----------------------------------------------------
        # ESTADO DEL MODELO
        # ----------------------------------------------------

        self.estado = QLabel(
            "Preparando modelo..."
        )

        self.estado.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.estado.setStyleSheet(
            "color: #64748B;"
        )

        layout.addWidget(
            self.estado
        )

        # Iniciar entrenamiento
        self.iniciar_entrenamiento()

    # ========================================================
    # 7. CARGAR O ENTRENAR MODELO
    # ========================================================

    def iniciar_entrenamiento(self):

        self.worker = Entrenamiento()

        self.worker.estado.connect(
            self.actualizar_estado
        )

        self.worker.terminado.connect(
            self.modelo_listo
        )

        self.worker.error.connect(
            self.mostrar_error
        )

        self.worker.start()

    def actualizar_estado(self, mensaje):

        self.estado.setText(mensaje)

    def modelo_listo(self, modelo, precision):

        self.modelo = modelo

        self.btn_reconocer.setEnabled(True)

        self.estado.setText(
            "Modelo listo. Puedes dibujar."
        )

        if precision >= 0:

            self.estado.setText(
                f"Modelo listo. Evaluacion: {precision:.2f}%"
            )

    def mostrar_error(self, mensaje):

        self.estado.setText(
            "Error al preparar el modelo."
        )

        self.btn_reconocer.setEnabled(False)

        QMessageBox.critical(
            self,
            "Error de Machine Learning",
            mensaje
        )

    # ========================================================
    # 8. RECONOCER NUMERO
    # ========================================================

    def reconocer(self):

        try:

            # Comprobar modelo
            if self.modelo is None:

                QMessageBox.warning(
                    self,
                    "Modelo no disponible",
                    "El modelo todavia no esta preparado."
                )

                return

            # Obtener imagen
            imagen = self.pizarra.obtener_imagen()

            # Preparar imagen
            datos = preparar_imagen(imagen)

            if datos is None:

                QMessageBox.warning(
                    self,
                    "Dibujo vacio",
                    "Primero debes dibujar un numero."
                )

                return

            # Verificar dimensiones
            print(
                "Dimensiones de entrada:",
                datos.shape
            )

            # Comprobar dimensiones esperadas
            if datos.shape != (1, 28, 28, 1):

                raise ValueError(
                    "La imagen no tiene las dimensiones correctas."
                )

            # PREDICCION DE LA RED NEURONAL
            probabilidades = self.modelo.predict(
                datos,
                verbose=0
            )[0]

            # Comprobar las diez categorias
            if len(probabilidades) != 10:

                raise ValueError(
                    "El modelo no genero diez resultados."
                )

            # Numero con mayor probabilidad
            numero = int(
                np.argmax(probabilidades)
            )

            confianza = float(
                probabilidades[numero] * 100
            )

            # Mostrar resultado
            self.resultado.setText(
                f"Numero reconocido: {numero}"
            )

            self.resultado.setStyleSheet(
                "color: #16803C;"
            )

            self.confianza.setText(
                f"Confianza: {confianza:.2f}%"
            )

            # Tres mejores posibilidades
            mejores = np.argsort(
                probabilidades
            )[::-1][:3]

            texto = "Otras posibilidades:\n"

            for n in mejores:

                texto += (
                    f"Numero {n}: "
                    f"{probabilidades[n] * 100:.2f}%\n"
                )

            self.posibilidades.setText(texto)

        except Exception as error:

            print(
                "ERROR DURANTE EL RECONOCIMIENTO:"
            )

            print(error)

            QMessageBox.critical(
                self,
                "Error de reconocimiento",
                f"No se pudo reconocer el numero:\n\n{error}"
            )

    # ========================================================
    # 9. BORRAR PIZARRA
    # ========================================================

    def borrar(self):

        self.pizarra.borrar()

        self.resultado.setText(
            "Numero reconocido: --"
        )

        self.resultado.setStyleSheet(
            "color: #1E293B;"
        )

        self.confianza.setText(
            "Confianza del modelo: --"
        )

        self.posibilidades.setText(
            "Esperando un dibujo..."
        )


# ============================================================
# 10. EJECUCION DEL PROGRAMA
# ============================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    ventana = Aplicacion()

    ventana.show()

    sys.exit(app.exec())
