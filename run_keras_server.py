from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array

from PIL import Image

import io
import numpy as np
from flask import Flask, request, jsonify


app = Flask(__name__)
model = None

def loadModel():
	global model
	model = ResNet50(weights="imagenet")


def prepareImage(image, target):
	if image.mode != "RGB":
		image = image.convert("RGB")

	# Resize input image & preprocess
	image = image.resize(target)
	image = img_to_array(image)
	# set 1st dim to samples
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	return image


@app.route("/predict", methods=["POST"])
def predict():

	# init return object
	data = {"success": False}
	if request.method=="POST":
		if request.files.get("image"):
			# read image in PIL format
			image = request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# call preprocessing image method
			image = prepareImage(image,target=(224,224))

			# classify image & init a list of predictions to be returned to user
			preds = model.predict(image)
			results = imagenet_utils.decode_predictions(preds)
			data["predictions"] = []

			# Add to the return object
			for (imagenetID, label, prob) in results[0]:
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)

			# change request status to success
			data["success"] =True

	return jsonify(data)			


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	loadModel()
	app.run()	







