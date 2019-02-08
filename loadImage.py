import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image


def loadLabels():
    import os
    import numpy as np
    path = "./data/labels"
    file = path + ".npy"
    if os.path.isfile(file):
        return np.load(file).item()
    
    print("téléchargement des labels")
    import requests
    labels_url = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
    response = requests.get(labels_url)  # Make an HTTP GET request and store the response.
    labels = {int(key): value for key, value in response.json().items()}
    print("sauvegarde dans: " + path)
    np.save(path, labels)
    return labels


def loadImage(path):
    # Now that we have an img, we need to preprocess it.
    img = Image.open(path)
    
    # We need to:
    #       * resize the img, it is pretty big (~1200x1200px).
    #       * normalize it, as noted in the PyTorch pretrained models doc,
    #         with, mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    #       * convert it to a PyTorch Tensor.
    # We can do all this preprocessing using a transform pipeline.
    img_size = 224  # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
    transform_pipeline = transforms.Compose([transforms.Resize(img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    
    img = transform_pipeline(img)
    
    # PyTorch pretrained models expect the Tensor dims to be (num input imgs, num color channels, height, width).
    # Currently however, we have (num color channels, height, width); let's fix this by inserting a new axis.
    #img = img.unsqueeze(0)  # Insert the new axis at index 0 i.e. in front of the other axes/dims.
    img = img[:3, :, :].unsqueeze(0)  #enlève le channel alpha
    
    # Now that we have preprocessed our img, we need to convert it into a
    # Variable; PyTorch models expect inputs to be Variables. A PyTorch Variable is a
    # wrapper around a PyTorch Tensor.
    return Variable(img)


def top(arr, n=5):
    import heapq
    return heapq.nlargest(n, range(len(arr)), lambda x: arr[x])


def main():
    import sys
    img = sys.argv[1] if len(sys.argv) >= 2 else "../AtlasNet/data/plane_input_demo.png"
    print("fichier: " + img)
    
    print("chargement modèle")
    vgg = models.vgg16(pretrained=True)  # This may take a few minutes.
    print(vgg)
    
    print("preprocessing")
    img = loadImage(img)
    
    classes = loadLabels()
    print("exécution forward")
    # Now let's load our model and get a prediction!
    prediction = vgg(img)  # Returns a Tensor of shape (batch, num class labels)
    print("prediction, type:", type(prediction), "shape:", prediction.shape)
    prediction = prediction.data.numpy()[0]
    
    #prediction = prediction.data.numpy().argmax()
    best = top(prediction, 10)
    print("best", best)
    
    print([(classes[b], prediction[b]) for b in best]) # Converts the index to a string using our labels dict
    
    for _ in range(5):
        print(top(vgg(img).data.numpy()[0], 10))


if __name__ == '__main__':
    main()
