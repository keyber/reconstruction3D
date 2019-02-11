import torchvision.models as models
import torch
from torch import nn
from sklearn import manifold
from sklearn import decomposition
import matplotlib.pyplot as plt
import os
import loadImage
import numpy as np


def tSNE(vectors, nCat):
    #from mpl_toolkits.mplot3d import Axes3D
    #ax = plt.axes()#projection='3d')
    print("tSNE  input dimension", len(vectors[0]))
    
    n_components = min(len(vectors), 50)
    
    pca = decomposition.PCA(n_components=n_components)
    
    principalComponent = pca.fit_transform([v.numpy() for v in vectors])
    
    print("après PCA", type(principalComponent), principalComponent.shape)
    
    colors = [(r/255, g/255, b/255) for r, g, b in
              [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
               (0, 0, 0),  (128, 128, 128), (255, 165, 0), (210, 180, 140),
               (192, 192, 192), (128, 0, 0), (0, 128, 0), (0, 0, 128), (165, 42, 42)]]
    perCat = len(vectors) // nCat
    
    for perplexity in [5, 15, 50]:
        for learnig_rate in [30, 100, 200]:
            plt.figure()
            tsne = manifold.TSNE(n_components=2, learning_rate=learnig_rate,
                                 perplexity=perplexity, n_iter=100000)
            Y = tsne.fit_transform(principalComponent)
            
            for cat in range(nCat):
                plt.scatter(Y[cat*perCat:(cat+1)*perCat, 0], Y[cat*perCat:(cat+1)*perCat, 1],
                            c=[colors[cat % len(colors)]])
    

def _genAll(root, id_category, nPerCat, nPerObj):
    """renvoie un vecteur d'images chargées dans l'ordre
    taille: id_category * nPerCat * nPerObj
    """
    res = []
    for cat in id_category:
        path = root + str(cat) + "/"
        for key in os.listdir(path)[:nPerCat]:
            sub_path = path + key + "/rendering/"
    
            with open(sub_path + "renderings.txt") as f:
                cpt = 0
                for line in f:
                    res.append(loadImage.loadImage(sub_path + line[:-1]))
                    cpt += 1
                    if cpt == nPerObj:
                        break
            
    return res  #torch.utils.data.DataLoader(res, nPerCat)


def _save_latent(latent, root, id_category, nPerCat, nPerObj):
    cpt=0
    for cat in id_category:
        path = root + "/" + str(cat)
        if not os.path.isdir(path):
            os.mkdir(path)
        for ind_obj in range(nPerCat):
            path2 = path + "/" + str(ind_obj)
            if not os.path.isdir(path2):
                os.mkdir(path2)
            for ind_view in range(nPerObj):
                path3 = path2 + "/" + str(ind_view)
                if not os.path.isfile(path3):
                    np.save(path3, latent[cpt])
                cpt+=1


def _load_latent(root, id_category, nPerCat, nPerObj):
    latent = []
    for cat in id_category:
        path = root + "/" + str(cat)
        for ind_obj in range(nPerCat):
            path2 = path + "/" + str(ind_obj)
            for ind_view in range(nPerObj):
                path3 = path2 + "/" + str(ind_view) + ".npy"
                latent.append(torch.tensor(np.load(path3)))
    return latent


def get_latent(chosenSubSet, nPerCat, nPerObj):
    #chargement de la liste des catégories définies pas imageNet
    root = "../AtlasNet/data/ShapeNetRendering/"
    id_category = []
    with open(root + "synsetoffset2category.txt") as f:
        for line in f:
            id_category.append(line.split()[1])
    
    #récupère les catégories des indices demandés
    if chosenSubSet:
        id_category = [id_category[i] for i in chosenSubSet]
    
    local_path = "./data/latentVectors"
    try:
        #chargement des vecteurs latents
        latentVectors = _load_latent(local_path, id_category, nPerCat, nPerObj)
    except FileNotFoundError as e:
        print("vecteurs latents non trouvés sur le disque:\n",e)
        print("sauvegarde à l'emplacement", local_path)
        
        #chargement des images
        images = _genAll(root, id_category, nPerCat, nPerObj)
        
        first = images[0]
        print("images:", (len(chosenSubSet), nPerCat, nPerObj), "=", len(images), type(first))
        
        #chargement du modèle vgg
        vgg = genModel()
        
        #calcul des vecteurs latents
        print("exécution forward")
        first = vgg(first).data.numpy()[0]
        
        print("vecteur latent:", type(first), first.shape)
        
        print("calcul en cours")
        latentVectors = [vgg(im)[0].reshape(-1) for im in images]
        
        print("sauvegarde")
        #sauvegarde des vecteurs latents
        _save_latent(latentVectors, local_path, id_category, nPerCat, nPerObj)
    
    return latentVectors


def genModel():
    """
    Une image est de taille (224,224,3) = 150528,
    le modèle la résume en un vecteur de dimensions (512, 7, 7) = 25 088"""
    vgg = models.vgg16(pretrained=True)
    vgg = vgg.features
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg


def _test():
    vgg = models.vgg16(pretrained=True)
    #récupère une image
    im = _genAll("../AtlasNet/data/ShapeNetRendering/", ["02691156"], 1, 1)[0]
    l = list(vgg.features.children()) + list(vgg.classifier.children())
    
    print("taille des couches")
    try:
        for i in range(1, len(l)):
            net = nn.Sequential(*l[:i])
            first = net(im).data.numpy()[0]
            print(i, first.shape)
    except RuntimeError as e:
        print(e)
    
    print("\n\n\n")


def _main():
    #_test()

    chosenSubSet = [0, 2, 7]
    n = len(chosenSubSet)
    nPerClass = 100
    nPerObj = 1
    latentCat = get_latent(chosenSubSet, nPerClass, nPerObj)
    
    tSNE(latentCat, n)
    
    plt.show()


if __name__ == '__main__':
    _main()
