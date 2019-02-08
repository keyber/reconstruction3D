import torchvision.models as models
from torch import nn
from sklearn import manifold
from sklearn import decomposition
import matplotlib.pyplot as plt
import os
import loadImage


def tSNE(vectors, nCat):
    #from mpl_toolkits.mplot3d import Axes3D
    #ax = plt.axes()#projection='3d')
    print("tSNE  input dimension", len(vectors[0]))
    n_components = min(len(vectors), 50)
    pca = decomposition.PCA(n_components=n_components)
    principalComponent = pca.fit_transform(vectors)
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
    

def genAll(root, id_category, nPerCat, nPerObj):
    """renvoie un vecteur d'images chargées dans l'ordre
    taille: id_category * nPerCat * nPerObj
    """
    res = []
    for cat in id_category:
        path = root + "ShapeNetRendering/" + str(cat) + "/"
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


def genModel():
    vgg = models.vgg16(pretrained=True)
    vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg


def genModel2():
    """
    -2 donne un vecteur latent de dimensions (3, 224, 224) = 150 528
    -1 donne un vecteur latent de dimensions (512, 7, 7)   =  25 088
    """
    vgg = models.vgg16(pretrained=True)
    vgg = nn.Sequential(*list(vgg.children())[:-1])
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg

def main():
    vgg = genModel2()
    
    root = "../AtlasNet/data/ShapeNet/"
    id_category = []
    with open(root + "synsetoffset2category.txt") as f:
        for line in f:
            id_category.append(line.split()[1])
    
    nCat = len(id_category)
    nPerCat = 10
    nPerObj = 4
    
    imagesCat = genAll(root, id_category, nPerCat, nPerObj)
    print("images:", (nCat, nPerCat, nPerObj), "=", len(imagesCat), type(imagesCat[0]))
    
    print("exécution forward")
    tmp = vgg(imagesCat[0]).data.numpy()[0]
    print("vecteur latent:", type(tmp), tmp.shape)
    
    latentCat = [vgg(i).data.numpy()[0].reshape((-1,)) for i in imagesCat]
    #latentCat = [vgg(i).data.numpy()[0][0].reshape((-1,)) for i in imagesCat]
    
    tSNE(latentCat, nCat)
    
    plt.show()


if __name__ == '__main__':
    main()
