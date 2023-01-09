import matplotlib.pyplot as plt
import requests
import json
import numpy as np
from tensorflow.keras.datasets import cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

#Normalizing data
x_train = x_train/255
x_test = x_test/255

#server URL
url = 'http://localhost:8501/v1/models/CIFAR_classifier:predict'

def make_prediction(instances):
   data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
   headers = {"content-type": "application/json"}
   json_response = requests.post(url, data=data, headers=headers)
   predictions = json.loads(json_response.text)['predictions']
   return predictions

predictions = make_prediction(x_test[0:4])

for i in range(4):
    print(f"Image sent to Tensorflow Serving was {y_test[i]}, it was predicted as {np.argmax(predictions[i])}")