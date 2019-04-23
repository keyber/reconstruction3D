import torchvision.models as models
import torch
import os
import numpy as np
import ply, loadImage


def _gen_latent(root, id_category, nPerCat, nPerObj):
    """renvoie un vecteur d'images chargées dans l'ordre
    taille: id_category * nPerCat * nPerObj"""
    res = []
    for cat in id_category:
        path = root + str(cat) + "/"
        for key in sorted(os.listdir(path))[:nPerCat]:
            sub_path = path + key + "/rendering/"
            
            with open(sub_path + "renderings.txt") as f:
                cpt = 0
                for line in f:
                    res.append(loadImage.loadImage(sub_path + line[:-1]))
                    cpt += 1
                    if cpt == nPerObj:
                        break
    
    return res  #torch.utils.data.DataLoader(res, nPerCat) todo


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


def _save_latent(latent, root, id_category, nPerCat, nPerObj):
    cpt = 0
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
                cpt += 1


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
    root = os.path.abspath(os.path.dirname(__file__)) + "/../../../AtlasNet/data/ShapeNetRendering/"
    id_category = _get_categories(root, chosenSubSet)
    local_path = os.path.abspath(os.path.dirname(__file__)) + "/../../data/latentVectors"
    
    try:
        #chargement des vecteurs latents
        latentVectors = _load_latent(local_path, id_category, nPerCat, nPerObj)
    except FileNotFoundError as e:
        print("vecteurs latents non trouvés sur le disque:\n", e)
        print("sauvegarde à l'emplacement", local_path)
        
        #chargement des images
        images = _gen_latent(root, id_category, nPerCat, nPerObj)
        
        first = images[0]
        print("images:", (len(chosenSubSet), nPerCat, nPerObj), "=", len(images), type(first))
        
        #chargement du modèle vgg
        vgg = gen_model()
        
        #calcul des vecteurs latents
        print("exécution forward")
        first = vgg(first).data.numpy()[0]
        
        print("vecteur latent:", type(first), first.shape)
        
        print("calcul en cours")
        latentVectors = [vgg(im)[0].reshape(-1) for im in images]
        
        print("sauvegarde")
        #sauvegarde des vecteurs latents
        _save_latent(latentVectors, local_path, id_category, nPerCat, nPerObj)
    
    assert not latentVectors[0].requires_grad
    return latentVectors


def get_clouds(chosenSubSet, nPerCat, size):
    """ratio int or float"""
    assert type(size) is float and 0 < size <= 1 or type(size) is int
    path = os.path.abspath(os.path.dirname(__file__)) + "/../../data/customShapeNet"
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


def gen_model():
    """Une image est de taille (224,224,3) = 150 528,
    le modèle la résume en un vecteur de dimensions (512, 7, 7) = 25 088"""
    vgg = models.vgg16(pretrained=True)
    vgg = vgg.features
    vgg.eval()
    for param in vgg.parameters():
        assert param.requires_grad
        param.requires_grad = False
    return vgg
