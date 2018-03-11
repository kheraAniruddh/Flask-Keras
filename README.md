# Flask-Keras
Wrapping keras' ResNet50 model with Flask API.
The data is imageNet ("preinstalled") with keras.


To make request use:
`curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict`
