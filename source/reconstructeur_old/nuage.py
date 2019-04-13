import numpy as np
import torch


class Nuage:
    """stocke l'ensemble des points sous la forme
    d'une liste de points et
    d'un tableau à 3 dimensions"""
    
    def __init__(self, points, segmentation):
        self.segmentation = segmentation
        self.liste_points = points
        
        # crée le tableau
        self.mat = segmentation.gen_matrix()
        
        # remplit le tableau
        for p in self.liste_points:
            x, y, z = self.segmentation.get_mat_coo(p)
            self.mat[x, y, z] = torch.cat((self.mat[x, y, z], p.unsqueeze(0)))
        
        self.sample_weights = torch.ones(len(self.liste_points))
    
    def recreate(self, points):
        """vide et re-rempli l'ancien tableau plutôt que de le supprimer et d'en allouer un nouveau"""
        self.liste_points = points
        
        # vide
        self.segmentation.filler(self.mat, self.mat)
        
        # remplit
        for p in self.liste_points:
            x, y, z = self.segmentation.get_mat_coo(p.detach())
            self.mat[x, y, z] = torch.cat((self.mat[x, y, z], p.unsqueeze(0)))
        
        return self
    
    def sub_sample(self, sample_size):
        return Nuage(self.liste_points[torch.multinomial(self.sample_weights, sample_size)], self.segmentation)
    
    # pour afficher des statistiques: liste du nombre de points parcourus pour chaque appel à get_closest
    points_parcourus = []
    
    def get_closest(self, point):
        # calcule la case dans laquelle tomberait le point
        case_coo = self.segmentation.get_mat_coo(point.detach().numpy())
        
        # le point le plus proche ne se trouve pas forcément dans la case la plus proche
        # un point plus proche peut encore être trouvé dans une case plus lointaine tant que
        # la distance à l'angle le plus proche de cette case est inférieur au min actuel
        point_coo = self.segmentation.get_float_coo(point.detach().numpy())
        
        cpt_point_parcourus = 0
        closest_point = torch.tensor(float("inf"))
        for d_2 in self.segmentation.distances_2:
            # stoppe la recherche si la distance minimale entre les angles des deux cases n'est pas meilleure
            if d_2 >= closest_point:
                break
            
            # considère toutes les cases non vides à distance d de notre case
            neighbours = [v for v in self.segmentation.get_neighbours(case_coo, d_2) if len(self.mat[v[0], v[1], v[2]])]
            if not len(neighbours): continue
            neighbours = np.array(neighbours)
            
            # calcule la distance entre notre point et le point des cases le plus proche
            cell_diff = point_coo - neighbours
            # on peut se déplacer dans la case en rajoutant un nombre compris entre 0 et 1
            cell_diff = cell_diff - np.minimum(1, np.maximum(0, cell_diff))
            cell_diff = np.sum(np.square(cell_diff), axis=1)
            # passe d'une distance entre cases à une distance entre point
            cell_diff = cell_diff * self.segmentation.factor_2
            
            # enlève les cases dont la distance min est trop grande
            ind_kept = np.where(cell_diff <= closest_point.item())
            if not len(ind_kept): continue
            cell_diff = cell_diff[ind_kept]
            neighbours = neighbours[ind_kept]
            
            # tri les cases par ordre croissant de distance
            ind_sorted = np.argsort(cell_diff)
            cell_diff = cell_diff[ind_sorted]
            neighbours = neighbours[ind_sorted]
            
            for real_d_2, v in zip(cell_diff, neighbours):
                if real_d_2 > closest_point:
                    break
                
                cpt_point_parcourus += len(self.mat[v[0], v[1], v[2]])
                
                # regarde si le point le plus proche parmi tous les points de la case est mieux
                candidat = torch.min(torch.sum(torch.pow(point - self.mat[v[0], v[1], v[2]], 2), dim=(1,)))
                if candidat < closest_point:
                    closest_point = candidat
            else:
                continue  # only executed if the inner loop did NOT break
            break  # only executed if the inner loop DID break
        
        Nuage.points_parcourus.append(cpt_point_parcourus)
        # return torch.sqrt(closest_point)
        return closest_point
    
    def chamfer_seg(self, other, k=1):
        k1 = 1 / (1 + k)
        k2 = k / (1 + k)
        # les points doivent bien être atteint par un MLP
        # noinspection PyTypeChecker
        loss0 = torch.sum(torch.cat([self.get_closest(p).unsqueeze(0) for p in other.liste_points]))
        loss0 *= k1 / len(other.liste_points)
        
        # les MLP doivent bien atteindre un point
        if k2 == 0:
            loss1 = 0
        else:
            # noinspection PyTypeChecker
            loss1 = torch.sum(torch.cat([other.get_closest(p).unsqueeze(0) for p in self.liste_points]))
            loss1 *= k2 / len(self.liste_points)
        
        return loss0, loss1
    
    warning = True
    
    @staticmethod
    def chamfer_quad(self, other, k=1):
        """return chamfer loss between
        the generated set Y = {f(x, p) for each f, p}
        and the real pointcloud S corresponding to the latent vector x

        sum_F(sum_A) et min_A(min_F) correspondent à une simple somme ou min sur l'ensemble des données
        donc on représente l'ensemble des points générés Y comme une liste et non comme une matrice
        """
        if type(self) == Nuage or type(other) == Nuage:
            if Nuage.warning:
                print("remplissage de la grille inutile")
                Nuage.warning = False
            self = self.liste_points
            other = other.liste_points
        k1 = 1 / (1 + k)
        k2 = k / (1 + k)

        # noinspection PyTypeChecker
        normes = torch.cat([torch.cat([torch.sum(torch.pow(y - s, 2)).unsqueeze(0) for s in other]) for y in self])
        normes = normes.reshape((len(self), len(other)))
        
        loss0 = torch.sum(torch.min(normes, dim=0)[0])
        loss0 *= k1 / len(other)
        
        if k2 == 0:
            loss1 = 0
        else:
            loss1 = torch.sum(torch.min(normes, dim=1)[0])
            loss1 *= k2 / len(self)
        return loss0, loss1
        
        # # On sert uniquement des min selon les deux axes, on peut ne pas stocker la matrice pour éviter les problèmes de RAM
        # # listes des minimums sur chaque ligne et colonne
        # min_axe0 = [torch.tensor(float("+inf"))] * len(S)
        # min_axe1 = [torch.tensor(float("+inf"))] * len(Y)
        # # pour chaque case de la matrice, met à jour les deux minimums correspondants
        # for i, s in enumerate(S):
        #     for j, y in enumerate(Y):
        #         val = torch.norm(y - s)
        #         min_axe0[i] = torch.min(val, min_axe0[i])
        #         min_axe1[j] = torch.min(val, min_axe1[j])
        # return torch.sum(min_axe0), torch.sum(min_axe1)
