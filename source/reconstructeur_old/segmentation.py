import numpy as np
import torch


class Segmentation:
    """défini la manière de partitionner l'espace 3D en cases"""
    
    def __init__(self, n_step):
        assert n_step % 2 == 1, "n_step doit être impair pour que le centre soit une cellule"
        self.dim = 3
        self.coo_min = -1
        self.coo_max = 1
        self.n_step = n_step
        self.n_step_div2 = n_step / 2
        
        # facteur pour passer de distances entre cases à des distances entre points
        self.factor_2 = (1 / self.n_step) ** 2
        
        # ensemble des déplacements(/voisins) possibles, ordonnés par leur distances
        self.neighbours, self.distances_2 = self._gen_neighbours()
        
        # sert à initialiser la matrice des nuages
        self.filler = np.frompyfunc(lambda x: torch.tensor([]), 1, 1)
    
    def _gen_neighbours(self):
        # génère tous les déplacements possibles de -n à n dans chaque dim
        r = range(self.n_step * 2)
        list_neighbours = np.array([[[(i, j, k) for i in r] for j in r] for k in r]).reshape(-1, 3)
        
        # shift de n pour centrer
        list_neighbours = list_neighbours - self.n_step
        
        # trie par distance au centre
        # il y a bcp d'équivalents
        # on regroupe les équivalents dans une même liste
        dict_neighbours = {}
        for voisin in list_neighbours:
            # la distance minimale entre des points des deux cases est égale à
            # la norme du vecteur entre les cases auquel on ENLEVE 1 DANS TOUTES LES DIMENSIONS NON NULLES
            # on laisse tout au carré
            d2 = np.sum(np.square(np.maximum(np.abs(voisin) - 1, 0)))
            
            # convertit la distance entre cases en une distance entre point
            d2 *= self.factor_2
            
            # une dimension égale à 1 devient équivalente à une dimension nulle
            # on ajoute un terme négligeable pour mettre en premiers les cases les plus proches du centre
            d2 += np.sum(np.abs(voisin) == 1) * 1e-6
            
            if d2 not in dict_neighbours:  # pas de pb d'arrrondi car on travaille sur des entiers, puis effectue les mêmes opérations
                dict_neighbours[d2] = []
            
            dict_neighbours[d2].append(voisin)
        
        dict_neighbours = {k: np.array(v) for (k, v) in dict_neighbours.items()}
        
        # calcule la liste triée des distances
        list_dist_2 = np.array(sorted(list(dict_neighbours.keys())))
        
        return dict_neighbours, list_dist_2
    
    def get_neighbours(self, cell, dist):
        """retourne la liste des cases à distance dist de cell"""
        # ajoute notre position pour calculer les coordonnées des voisins
        res = cell + self.neighbours[dist]
        
        # filtre les voisins qui sortent du tableau
        return res[np.all((0 <= res) & (res < self.n_step), axis=1)]
    
    def gen_matrix(self):
        """retourne un tableau 3D servant à stocker les points"""
        a = np.empty((self.n_step, self.n_step, self.n_step), dtype=list)
        return self.filler(a, a)
    
    def get_mat_coo(self, point):
        """retourne les coordonnées de la case du tableau contenant ce point"""
        res = np.asarray((point + 1) * self.n_step_div2, int)
        #<=> int((coo - self.coo_min) / (self.coo_max - self.coo_min) * self.n_step)
        
        # si une coordonnée est maximale (càd 1), le point sort du tableau, on le met dans la case d'avant
        return np.minimum(res, self.n_step - 1)
    
    def get_float_coo(self, point):
        """retourne les coordonnées de notre point dans la base des cases du tableau"""
        return np.minimum((point + 1) * self.n_step_div2, self.n_step - 1)
