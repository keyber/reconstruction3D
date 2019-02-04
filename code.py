import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn import manifold, datasets
import matplotlib.pyplot as plt

def tSNE():
    n_samples = 300
    n_components = 2
    (fig, subplots) = plt.subplots(3, 5, figsize=(15, 8))
    perplexities = [5, 30, 50, 100]
    
    X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    red = y == 0
    green = y == 1
    
    ax = subplots[0][0]
    ax.scatter(X[red, 0], X[red, 1], c="r")
    ax.scatter(X[green, 0], X[green, 1], c="g")
    plt.axis('tight')
    
    for i, perplexity in enumerate(perplexities):
        ax = subplots[0][i + 1]
    
        tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(X)
        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(Y[red, 0], Y[red, 1], c="r")
        ax.scatter(Y[green, 0], Y[green, 1], c="g")
        ax.axis('tight')
    plt.show()

def normalize(args):
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
resnet18 = models.resnet18(pretrained=True)
print(resnet18)
resnet18()