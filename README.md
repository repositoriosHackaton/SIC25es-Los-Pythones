# SIC25es-Los-Pythones

## Proyecto de Predicción de Gastos Mensuales

Este proyecto tiene como objetivo predecir los gastos mensuales de una persona con base en sus características como edad, salario, dependientes, educación, vivienda y región.

## Estructura del Proyecto

- `data/`: Contiene los archivos de datos.
- `models/`: Contiene los modelos entrenados y los archivos de escaladores.
- `src/`: Contiene el código fuente del proyecto (preprocesamiento, entrenamiento, predicción, etc.).
- `templates/`: Contiene las plantillas HTML para la interfaz web.
- `uploads/`: Aquí se almacenan los archivos subidos por los usuarios.
- `venv/`: Contiene el entorno virtual del proyecto.
- `requirements.txt`: Contiene las dependencias necesarias para ejecutar el proyecto.

```
proyecto_prediccion_gastos/
│── data/
│   ├── dataset_gastos_mensuales.csv
│
│── models/
│   ├── modelo_gastos.pth
│   ├── scaler.pkl
│
│── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── app.py
│
│── templates/
│   ├── index.html
│   ├── analysis.html
│
│── uploads/
│
│── venv/
│
── requirements.txt
```

## Instalación

1. Crear un entorno virtual:

   ```bash
   python -m venv venv
   ```

2. Activar el entorno virtual:
   
   - En Windows:
     ```bash
     venv\Scripts\activate
     ```
   - En Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

3. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Ejecutar la aplicación:

   ```bash
   python src/app.py
   ```

## Uso

1. Abre el navegador y ve a `http://127.0.0.1:5000/`.
2. Ingresa los datos solicitados (Edad, Salario, Dependientes, Educación, Vivienda y Región) y obtendrás una predicción de tus gastos mensuales.

## Descripción de los Datos

El modelo utiliza las siguientes características para predecir los gastos mensuales:

- **Edad**: Edad de la persona (mayor o igual a 18 años).
- **Salario**: Salario mensual neto en USD (mayor o igual a 365 USD mensuales).
- **Dependientes**: Número de personas a cargo económicamente.
- **Educación**: Nivel educativo (categorizado numéricamente):
  - 1 = Primaria
  - 2 = Secundaria
  - 3 = Universitaria
  - 4 = Postgrado
- **Vivienda**: Tipo de vivienda (categorizado numéricamente):
  - 1 = Propia
  - 2 = Alquilada
  - 3 = Hipotecada
- **Región**: Región geográfica (categorizado numéricamente):
  - 1 = Urbana
  - 2 = Rural

## Contribuciones

Si deseas contribuir al proyecto, por favor realiza un fork y envía un pull request con tus mejoras.

---

¡Gracias por usar el Proyecto de Predicción de Gastos Mensuales! ¡Esperamos que encuentres útil esta herramienta para predecir tus gastos mensuales!
```
