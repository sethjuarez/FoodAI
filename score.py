import json
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import onnxruntime as rt
from torchvision import transforms

# azureml imports
from azureml.core.model import Model

def init():
    global session, transform, classes, input_name

    try:
        model_path = Model.get_model_path('FoodAI')
    except:
        model_path = 'model.onnx'

    classes = ['hot_dog', 'pizza']
    session = rt.InferenceSession(model_path) 
    input_name = session.get_inputs()[0].name
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def run(raw_data):

    post = json.loads(raw_data)
    image_url = post['image']
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    v = transform(img)
    pred_onnx = session.run(None, {input_name: v.unsqueeze(0).numpy()})
    return classes[np.argmax(pred_onnx)]

if __name__ == '__main__':
    init()
    pizza = 'https://images-gmi-pmc.edge-generalmills.com/f4c0a86f-b080-45cd-a8a7-06b63cdb4671.jpg'
    hot_dog = 'https://leitesculinaria.com/wp-content/uploads/fly-images/96169/best-hot-dog-recipe-fi-400x225-c.jpg'

    print('\n---------------------\nInference with pizza:')
    print(run(json.dumps({'image': pizza})))
    print('\nInference with hot dog:')
    print(run(json.dumps({'image': hot_dog})))