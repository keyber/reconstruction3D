import torchvision.models as models
import torch
import os
import numpy as np
import ply


def _gen_clouds(root, id_category, nPerCat, sous_echantillonage):
    root += "/"
    res = []
    for cat in id_category:
        path = root + str(cat) + "/ply/"
        # ignore les fichiers .txt
        keys = [key for key in os.listdir(path) if key[-4:]!=".txt"]
        for key in sorted(keys)[:nPerCat]:
            sub_path = path + key
            cloud = ply.read_ply(sub_path)['points']
            
            if type(sous_echantillonage) == int:
                ratio = sous_echantillonage / len(cloud.values)
                if ratio > 1:
                    print("PAS ASSEZ DE POINTS DANS LE NUAGE", sous_echantillonage, '/', len(cloud.values))
            else:
                ratio = sous_echantillonage
            
            sub_sampled = []
            for i, x in enumerate(cloud.values[:, :3]):
                if len(sub_sampled) / (i + 1) < ratio:
                    sub_sampled.append(torch.tensor(x))
            
            assert ratio>1 and len(sub_sampled) == len(cloud.values) or len(sub_sampled) == ratio * len(cloud.values)
            # noinspection PyTypeChecker
            res.append(torch.cat(sub_sampled).reshape((-1, 3)))
    
    assert not res[0].requires_grad
    return np.array(res, dtype=torch.Tensor)



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


def _get_categories(path, chosenSubSet):
    path += "/synsetoffset2category.txt"
    
    id_category = []
    with open(path) as f:
        for line in f:
            id_category.append(line.split()[1])
    
    #récupère les catégories des indices demandés
    if chosenSubSet:
        id_category = [id_category[i] for i in chosenSubSet]
    return id_category


def get_latent(chosenSubSet, nPerCat, nPerObj):
    id_category = _get_categories("drive/Colab/data/ShapeNetRendering/", chosenSubSet)
    latentVectors = _load_latent("drive/Colab/data/latentVectors", id_category, nPerCat, nPerObj)
    
    assert not latentVectors[0].requires_grad
    return latentVectors


def get_clouds(chosenSubSet, nPerCat, size):
    """ratio int or float
    ratio > nombre de points dans le fichier => ratio=1 (+ warning)"""
    assert type(size) is float and 0 < size <= 1 or type(size) is int
    path = "drive/Colab/data/customShapeNet"
    id_category = _get_categories(path, chosenSubSet)
    return _gen_clouds(path, id_category, nPerCat, size)


def write_clouds(path, clouds):
    import pandas
    if os.path.isdir(path):
        print(path, "écrasé")
    else:
        print(path, "écrit")
        os.mkdir(path)
    path += '/'
    for i, c in enumerate(clouds):
        ply.write_ply(path + str(i), pandas.DataFrame(c), as_text=True)
