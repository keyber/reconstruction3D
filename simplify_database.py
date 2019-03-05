import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--ply', type=str, help="chemin vers la racine du dossier contenant les nuages de points")
parser.add_argument('--png', type=str, help="chemin vers la racine du dossier contenant les images 2D")
parser.add_argument('--intersect', type=bool, help="pour chaque dossier ply, ne garde que les png correspondant à un nuage de points", default=False)
parser.add_argument('--per_cat', type=int, help="nombre d'objet à garder de chaque catégorie (.ply ou dossier de .png)", default=None)
parser.add_argument('--per_obj', type=int, help="nombre de vue (.png) à garder pour chaque objet dossier", default=None)
parser.add_argument('--rm_other', type=bool, help="supprime les fichiers '*.points.ply', .ply2.txt renderings.txt rendering_metadata.txt", default=False)
params = parser.parse_args()
ply = params.ply
png = params.png
intersect = params.intersect
per_cat = params.per_cat
per_obj = params.per_obj
rm_other = params.rm_other

print("Cet outil supprime des images et des nuages de points pour ne garder que ceux souhaités")
print(vars(params))

ply = "../AtlasNet/data/customShapeNet"
#png = "../AtlasNet/data/ShapeNetRendering2"
rm_other = 1
#intersect = 1
per_cat = 500
per_obj = 10


if not ply and not png:
    raise ValueError("pour faire quelque chose il faut au moins un dossier")

if ply and not os.path.isdir(ply) or png and not os.path.isdir(png):
    raise FileNotFoundError()

if per_cat is not None and per_cat < 10 or per_obj is not None and per_obj < 1:
    raise ValueError("On va supprimer beaucoup trop, modifiez ce fichier si vous le voulez vraiment")

if intersect and not (ply and png):
    raise ValueError("pour faire l'intersection, il faut les deux dossiers")


print("dossiers trouvés :")
for path in [ply, png]:
    if path:
        print(path,":")
        for category in os.listdir(path):
            if os.path.isdir(os.path.join(path, category)):
                print(category, end=' ')
        print("\n")


print("continuer ? y/n")
if input() != "y":
    exit()


if ply:
    if intersect:
        objets = {}
    
    for category in os.listdir(ply):
        path_category = os.path.join(ply, category, "ply")
        if os.path.isdir(path_category):
            list_objets = []
            
            # construit la liste des objets utiles
            # enlève les autres fichiers au passage si demandé
            for x in os.listdir(path_category):
                if x[-5:] == "2.txt" or x == "*.points.ply":
                    if rm_other:
                        os.remove(os.path.join(path_category, x))
                    continue
                
                list_objets.append(x)
            
            # enlève le surplus
            if per_cat and len(list_objets) > per_cat:
                list_objets = sorted(list_objets)
                for x in list_objets[per_cat:]:
                    os.remove(os.path.join(path_category, x))
                list_objets = list_objets[:per_cat]
            
            if intersect:
                # pour chaque catégorie, stocke l'ensemble des noms de fichiers
                # enlève le .points.ply
                objets[category] = set([x[:-11] for x in list_objets])

if png:
    def rm_dir_png(path):
        for fichier in os.listdir(os.path.join(path, "rendering")):
            os.remove(os.path.join(path, "rendering", fichier))
        os.rmdir(os.path.join(path, "rendering"))
        os.rmdir(os.path.join(path))


    for category in os.listdir(png):
        path_category = os.path.join(png, category)
        if os.path.isdir(path_category):
            list_objets = []
            
            # construit la liste des objets
            # enlève les png qui ne correspondent pas à un nuage
            for x in os.listdir(path_category):
                if intersect and category in objets and x not in objets[category]:
                    rm_dir_png(os.path.join(path_category, x))
                    continue
                
                list_objets.append(x)
                
            # enlève le surplus d'objets
            if per_cat and len(list_objets) > per_cat:
                list_objets = sorted(list_objets)
                for x in list_objets[per_cat:]:
                    rm_dir_png(os.path.join(path_category, x))
                list_objets = list_objets[:per_cat]
            
            # enlève le surplus de vues
            # enlève les fichiers inutiles
            if per_obj or rm_other:
                for x in list_objets:
                    for file in os.listdir(os.path.join(path_category, x, "rendering")):
                        if file[-4:] == ".png":
                            if per_obj and int(file[:2]) >= per_obj:
                                os.remove(os.path.join(path_category, x, "rendering", file))
                        elif rm_other:
                            os.remove(os.path.join(path_category, x, "rendering", file))
    