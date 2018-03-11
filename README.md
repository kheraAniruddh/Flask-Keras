# Flask-Keras
Wrapping keras' ResNet50 model with Flask API.
The data is imageNet ("preinstalled") with keras.


To make request use:
`curl -X POST -F image=@pingpong.jpg 'http://localhost:5000/predict`

 Referred: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
