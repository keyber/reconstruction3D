import torchvision.models as models
from sklearn import manifold
import matplotlib.pyplot as plt
import matplotlib.colors as color
import os
import demo2


def tSNE(vectors, nCat, perplexity=50):
    tsne = manifold.TSNE(n_components=2, init='random',
                         random_state=0, perplexity=perplexity)
    Y = tsne.fit_transform(vectors)
    #colors="rgbycmk"
    perCat = len(vectors) // nCat
    for cat in range(nCat):
        plt.scatter(Y[cat*perCat:(cat+1)*perCat, 0], Y[cat*perCat:(cat+1)*perCat, 1],
                    color=color.hsv_to_rgb((cat / (nCat-1), 1, 1)))
    

def genAllViewsFirstObject(root, id0):
    """toutes les vues du premier objet de la catégorie"""
    path = root + "ShapeNetRendering/" + str(id0) + "/"
    path = path + os.listdir(path)[0] + "/rendering/"
    res = []
    with open(path + "renderings.txt") as f:
        for line in f:
            res.append(demo2.loadImage(path + line[:-1]))
    
    return res


def genAllCategories(root, id_category, nPerCat):
    """la premiere vue de chaque objet"""
    res = []
    for cat in id_category:
        path = root + "ShapeNetRendering/" + str(cat) + "/"
        for key in os.listdir(path)[:nPerCat]:
            sub_path = path + key + "/rendering/00.png"
            res.append(demo2.loadImage(sub_path))
    
    return res #torch.utils.data.DataLoader(res, nPerCat)


def main():
    vgg = models.vgg16(pretrained=True)
    
    root = "../AtlasNet/data/ShapeNet/"
    id_category = []
    with open(root + "synsetoffset2category.txt") as f:
        for line in f:
            id_category.append(line.split()[1])
    
    nPerCat = 10
    nCat = len(id_category)
    
    #imagesView = genAllViewsFirstObject(root, id_category[0])
    imagesCat = genAllCategories(root, id_category, nPerCat)
    print(len(imagesCat))
    print(type(imagesCat[0]))
    
    # 10 fois la meme image
    #imagesSame = [imagesView[0]]*10
    
    print("exécutions forward sur la partie feature")
    
    #latentSame = [vgg.features(i).data.numpy()[0].reshape((512*7*7, )) for i in imagesSame]
    #latentView = [vgg.features(i).data.numpy()[0].reshape((512*7*7, )) for i in imagesView]
    latentCat = [vgg.features(i).data.numpy()[0].reshape((512*7*7, )) for i in imagesCat]
    print(len(latentCat))
    print(latentCat[0].shape)
    
    tSNE(latentCat, nCat)
    
    plt.show()


if __name__ == '__main__':
    main()
