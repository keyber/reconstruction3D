import torch
import scipy.spatial
from scipy.spatial import cKDTree as KDTree
# from scipy.spatial import KDTree
import sys
sys.setrecursionlimit(10000) # 1000 par défaut
# la construction des kd-tree est récursive et plante si trop de points
# 2^15 > 30000 donc une depth de 15 devrait suffire théoriquement
# les cKDTree sont assez bien équilibrés la profondeur ne dépasse pas 500
# les KDTree ne sont pas assez bien équilibrés (atteignent 2000)


class Nuage:
    """
    La complexité du calcul de la distance de chamfer est
        |y| * 1NN(ŷ)  +  |ŷ| * 1NN(y)
    L'algorithme naïf parcourant tout le nuage pour trouver le plus proche voisin a une complexité quadratique :
        2 * |y| * |ŷ|
    En utilisant une structure plus appropriée comme un kd-tree on peut obtenir :
        |y| * log(|ŷ|)  +  |ŷ| * log(|y|)
    Les nuages de ground truth de 30.000 sont sous-échantillonnés.
    Afin de ne pas sur-apprendre des petits nuages, Groueix rééchantillonne les ground truth à chaque epoch.
    Re-créer les structures de données à chaque itération est coûteux.
    
    On va précalculer les kd-tree des ground truth
    """
    
    def __init__(self, points, eps):
        self.points = points # la liste des points sous forme de tensor
        self.kdtree = KDTree(points.detach().numpy()) # kdtree.data est forcément un numpy.array
        self.eps = eps
        
        # print(get_depth_KDTree(self.kdtree.tree))
    
    def chamfer(self, other, sub_sampling=None, k=1):
        """sub_sampling de la liste des points de other
        k: facteur entre les deux termes de la loss"""
        # todo on pourrait retourner un point parmis les (len / sub_sampling) plus proches voisins
        self_list = self.points
        other_list_full = other.points
        other_list_subsampled = other_list_full[sub_sampling] if sub_sampling is not None else other_list_full
            
        k1 = 1 / (1 + k)
        k2 = k / (1 + k)
        
        # les points doivent bien être atteint par un MLP
        # pour chaque point de other, récupère le point le plus proche dans self
        list_ind = [self.kdtree.query(p, eps=self.eps)[1] for p in other_list_subsampled] #query retourne (d_min, argmin)
        corresponding =  self_list[list_ind]
        diff = other_list_subsampled - corresponding
        # noinspection PyTypeChecker
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        loss0 = torch.mean(dist)
        # print("  mean", loss0.item(), "min", torch.min(dist).item(), "max", torch.max(dist).item(), "std", torch.std(dist).item())
        loss0 *= k1
        
        # les MLP ne doivent pas dépasser la zone à couvrir
        if k2 == 0:
            loss1 = torch.tensor(0.0)
        else:
            list_ind = [other.kdtree.query(p, eps=other.eps)[1] for p in self.kdtree.data]
            corresponding = other_list_full[list_ind]
            diff = self_list - corresponding
            # noinspection PyTypeChecker
            dist = torch.sum(torch.pow(diff, 2), dim=1)
            loss1 = torch.mean(dist)
            # print("  mean", loss1.item(), "min", torch.min(dist).item(), "max", torch.max(dist).item(), "std", torch.std(dist).item())
            loss1 *= k2
        
        return loss0, loss1

    def chamfer_quadratic(self, other, **kwargs):
        del kwargs #enlève le warning (kwargs récupère d'autres params spécifiés sans lever d'erreur)
        # noinspection PyTypeChecker
        normes = torch.cat([torch.cat([torch.sum(torch.pow(y - s, 2)).unsqueeze(0) for s in other.points]) for y in self.points])
        normes = normes.reshape((len(self.kdtree.data), len(other.kdtree.data)))
        # noinspection PyTypeChecker
        return torch.mean(torch.min(normes, dim=0)[0])*.5, torch.mean(torch.min(normes, dim=1)[0])*.5
    
    
def get_depth_KDTree(node, depth=0):
    if isinstance(node, scipy.spatial.KDTree.leafnode):
        return depth
    return max(get_depth_KDTree(node.less, depth+1), get_depth_KDTree(node.greater, depth+1))
    
    
def get_depth_cKDTree(node, depth=0):
    if node is None:
        return depth
    return max(get_depth_cKDTree(node.lesser, depth+1), get_depth_cKDTree(node.greater, depth+1))
