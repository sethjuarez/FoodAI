import json
import time
import requests
import datetime
import numpy as np
from PIL import Image
from io import BytesIO
import onnxruntime as rt
from torchvision import transforms

# azureml imports
from azureml.core.model import Model

session, transform, classes, input_name = None, None, None, None


def init():
    global session, transform, classes, input_name

    try:
        model_path = Model.get_model_path('foodai')
    except:
        model_path = '../outputs/model.onnx'

    classes = ['burrito', 'tacos']
    session = rt.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def run(raw_data):
    prev_time = time.time()

    post = json.loads(raw_data)
    image_url = post['image']
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    v = transform(img)
    pred_onnx = session.run(None, {input_name: v.unsqueeze(0).numpy()})[0][0]

    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)

    predictions = {}
    for i in range(len(classes)):
        predictions[classes[i]] = str(pred_onnx[i])

    payload = {
        'time': str(inference_time.total_seconds()),
        'prediction': classes[int(np.argmax(pred_onnx))],
        'scores': predictions
    }

    print('Input ({}), Prediction ({})'.format(post['image'], payload))

    return payload


if __name__ == '__main__':
    init()
    burrito = 'https://www.exploreveg.org/files/2015/05/sofritas-burrito.jpeg'
    tacos = 'https://upload.wikimedia.org/wikipedia/commons/2/27/Mmm..._Tacos.jpg'

    print('---------------------Inference with burrito:')
    print(run(json.dumps({'image': burrito})))
    print('Inference with taco:')
    print(run(json.dumps({'image': tacos})))
