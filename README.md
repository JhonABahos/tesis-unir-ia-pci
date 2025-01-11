# Proyecto de Tesis: Detección de Daños y Cálculo de PCI en Pavimentos

Este documento proporciona una guía detallada sobre la estructura de los módulos y archivos del proyecto, explicando la funcionalidad de cada uno y el flujo de ejecución principal.

---

## **Organización de carpetas y módulos**

### **`data_processing/`**
Agrupa los módulos encargados de la carga, preprocesamiento y manipulación de datos, generación y verificación de máscaras.

- **`dataset.py`**: Módulo encargado de la carga y manejo de los datos de entrenamiento y validación.
- **`preprocessing.py`**: Realiza las transformaciones y normalización de los datos de entrada.
- **`split_dataset.py`**: Divide el dataset en subconjuntos de entrenamiento y validación.
- **`mask_generator.py`**: Genera las máscaras de segmentación basadas en los datos de entrada.
- **`postprocess_masks.py`**: Realiza el postprocesamiento de las máscaras segmentadas para optimizar los resultados.
- **`check_masks_utils.py`**: Proporciona utilidades para verificar la validez y consistencia de las máscaras generadas.

---

### **`utils/`**
Módulos utilitarios para cálculos de soporte y generación de reportes.

- **`damages.py`**: Define los tipos de daños y gestiona su información relevante.
- **`deduct_values.py`**: Calcula los valores deducidos necesarios para la evaluación del índice de condición del pavimento (PCI).
- **`file_utils.py`**: Contiene funciones para la manipulación de archivos y rutas del proyecto.
- **`pci_calculator.py`**: Implementa el algoritmo de cálculo del PCI basado en los daños detectados.
- **`gpt4all_report.py`**: Genera reportes automáticos utilizando GPT4All para explicar los resultados obtenidos.

---

### **`models/`**
Módulos relacionados con la definición, entrenamiento y evaluación del modelo de IA.

- **`models.py`**: Contiene la definición del modelo DeepLabV3 utilizado para la segmentación de daños en pavimentos.
- **`training.py`**: Implementa el entrenamiento del modelo, incluyendo la definición de épocas, optimizador y pérdida.
- **`inference.py`**: Realiza las predicciones y evaluación del modelo entrenado sobre nuevos datos.

---

### **`training_pipeline.py`**
Este archivo sirve como punto de entrada principal para entrenar y validar el modelo. Se encarga de orquestar la configuración, carga de datos, entrenamiento y evaluación del modelo, centralizando el flujo de trabajo del proyecto.

---

## **Instrucciones de uso**

1. **Configuración:** Verifica los parámetros en `config.py` (ubicación de los datos, hiperparámetros de entrenamiento, etc.).
2. **Preprocesamiento de datos:** Ejecuta el módulo `data_processing/preprocessing.py` para normalizar y preparar las imágenes y máscaras.
3. **Entrenamiento:** Ejecuta `training_pipeline.py` para iniciar el proceso de entrenamiento y validación.
4. **Evaluación:** Utiliza `models/inference.py` para realizar predicciones y evaluar los resultados.
5. **Generación de reportes:** Corre `utils/gpt4all_report.py` para generar un reporte automático de los resultados.

---

## **Flujo de ejecución recomendado**

1. `config.py`: Configuración inicial del proyecto.
2. `data_processing/dataset.py`: Carga de datos.
3. `data_processing/preprocessing.py`: Preprocesamiento de datos.
4. `models/training.py`: Entrenamiento del modelo.
5. `models/inference.py`: Evaluación y predicciones.
6. `utils/pci_calculator.py`: Cálculo del índice PCI.
7. `utils/gpt4all_report.py`: Generación de reportes.

---

Esta organización te permite mantener un flujo de trabajo claro y modular, facilitando la extensión y mantenimiento del proyecto.

