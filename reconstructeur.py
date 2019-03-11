import torch
from torch import nn
import numpy as np
import tSNE
import time


class Segmentation:
    """défini la manière de partitionner l'espace 3D en cases"""
    
    def __init__(self, n_step):
        assert n_step % 2 == 1, "n_step doit être impair pour que le centre soit une cellule"
        self.dim = 3
        self.coo_min = -1
        self.coo_max = 1
        self.n_step = n_step
        self.n_step_2 = n_step / 2
        
        self.neighbours, self.distances_2 = self._gen_neighbours()
        # for d in self.distances_2:
        #     print(d,":",self.neighbours[d])
    
    def _gen_neighbours(self):
        # génère tous les déplacements possibles de -n à n dans chaque dim
        r = range(self.n_step * 2)
        list_neighbours = np.array([[[(i, j, k) for i in r] for j in r] for k in r])
        
        # shift de n pour centrer
        list_neighbours = [voisin - self.n_step for voisin in list_neighbours.reshape(-1, 3)]
        
        # trie par distance au centre
        # il y a bcp d'équivalents
        # on regroupe les équivalents dans une même liste
        dict_neighbours = {}
        factor_2 = ((self.coo_max - self.coo_min) / self.n_step) ** 2
        for voisin in list_neighbours:
            # la distance minimale entre des points des deux cases est égale à
            # la norme du vecteur entre les cases auquel on enlève 1 dans toutes les dimensions non nulles
            # on scale la distance pour passer de distances entre dans la base des cases à des distances entre points
            # on laisse tout au carré
            d2 = np.sum(np.power(np.maximum(np.abs(voisin) - 1, 0), 2)) * factor_2
            
            # une dimension égale à 1 devient équivalente à une dimension nulle
            # on ajoute un terme négligeable pour mettre en premiers les cases plus proches du centre
            d2 += (np.abs(voisin) == 1).sum() * 1e-6
            
            if d2 not in dict_neighbours:  # pas de pb d'arrrondi car on travaille sur des entiers, puis effectue les mêmes opérations
                dict_neighbours[d2] = []
            
            voisin = torch.tensor(voisin)
            voisin.requires_grad = False
            dict_neighbours[d2].append(voisin)
        
        # calcule la liste triée des distances
        list_dist_2 = np.array(sorted(list(dict_neighbours.keys())))
        
        return dict_neighbours, list_dist_2
    
    def get_neighbours(self, cell, dist):
        res = [v.numpy() + cell for v in self.neighbours[dist]]
        return [v for v in res if np.all((0 <= v) & (v < self.n_step))]
    
    def gen_matrix(self):
        filler = np.frompyfunc(lambda x: list(), 1, 1)
        a = np.empty((self.n_step, self.n_step, self.n_step), dtype=list)
        return filler(a, a)
    
    def get_mat_coo(self, point):
        # res = [int((coo - self.coo_min) / (self.coo_max - self.coo_min) * self.n_step) for coo in point]
        res = [int((coo + 1) * self.n_step_2) for coo in point]
        
        # si une coordonnée est maximale (càd 1), le point sort du tableau, on le met dans la case d'avant
        return [min(x, self.n_step - 1) for x in res]
    
    def get_float_coo(self, point):
        return np.minimum((point + 1) * self.n_step_2, self.n_step - 1)


class Nuage:
    """stocke l'ensemble des points sous la forme
    d'une liste de points et
    d'un tableau à 3 dimensions"""
    
    points_parcourus = []
    max_list = []
    
    def __init__(self, points, segmentation):
        self.segmentation = segmentation
        self.liste_points = points
        self.mat = segmentation.gen_matrix()
        
        for p in self.liste_points:
            x, y, z = self.segmentation.get_mat_coo(p)
            self.mat[x, y, z].append(p)
        
        self.filler = np.frompyfunc(lambda x: list(), 1, 1)
    
    def recreate(self, points):
        """vide et re-rempli l'ancien tableau plutôt que de le supprimer et d'en allouer un nouveau"""
        self.filler(self.mat, self.mat)
        
        self.liste_points = points
        
        for p in self.liste_points:
            x, y, z = self.segmentation.get_mat_coo(p)
            self.mat[x, y, z].append(p)
        
        return self
    
    def get_closest(self, point):
        # calcule la case dans laquelle tomberait le point
        case_coo = self.segmentation.get_mat_coo(point)
        
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
            neighbours = [v for v in self.segmentation.get_neighbours(case_coo, d_2) if self.mat[v[0], v[1], v[2]]]
            
            # calcule la distance entre notre point et l'angle le plus proche de la case
            precise_dist_2 = [(np.sum(np.power(point_coo - v, 2)), v) for v in neighbours]
            
            # enlève les cases dont la distance min est trop grande
            precise_dist_2 = [(d_2, v) for (d_2, v) in precise_dist_2 if d_2 < closest_point]
            
            # parcourt les cases par ordre croissant (pour réduire l'espérance de temps pour trouver le point le plus proche)
            for real_d_2, v in sorted(precise_dist_2, key=lambda x: x[0]):
                if real_d_2 > closest_point:
                    break
                cpt_point_parcourus += len(self.mat[v[0], v[1], v[2]])
                # regarde si le point le plus proche parmi tous les points de la case est mieux
                candidat = min(torch.sum(torch.pow(point - p, 2), dim=(0,)) for p in self.mat[v[0], v[1], v[2]])
                if candidat < closest_point:
                    closest_point = candidat
            
            else:
                continue  # only executed if the inner loop did NOT break
            break  # only executed if the inner loop DID break
        
        Nuage.points_parcourus.append(cpt_point_parcourus)
        # return torch.sqrt(closest_point)
        return closest_point
    
    def chamfer(self, other):
        loss0 = sum(self.get_closest(p) for p in other.liste_points)
        loss1 = sum(other.get_closest(p) for p in self.liste_points)
        return loss0, loss1


class Reconstructeur(nn.Module):
    def __init__(self, n_mlp, latent_size, n_grid_step, segmentation):
        super().__init__()
        print("nombre de MLP:", n_mlp)
        print("résolution de la grille:", n_grid_step, "^2 =", n_grid_step ** 2)
        print("taille des nuages générés:", n_mlp, "*", n_grid_step ** 2, "=", n_mlp * n_grid_step ** 2)
        print("résolution segmentation:", segmentation.n_step, "^3 =", segmentation.n_step ** 3)
        self._nuage_tmp = Nuage([], segmentation)
        self.input_dim = 2
        self.output_dim = 3
        self.n_grid_step = n_grid_step
        self.verbose = True
        
        self.decodeurs = nn.ModuleList([nn.Sequential(
            #on rajoute les coordonnées d'un point dans l'espace de départ
            nn.Linear(latent_size + self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            #on retourne un point dans l'espace de sortie
            nn.Linear(128, self.output_dim),
            #tanh ramène dans [-1, 1], comme les nuages de notre auto-encodeur et de groueix
            nn.Tanh()
        ) for _ in range(n_mlp)])
        
        self.grid = self._gen_grid()
        self.loss = Nuage.chamfer
    
    def _gen_grid(self):
        """retourne une liste de n points 2D au bon format :
        array(tensor(coo_x, coo_y))
        les points forment un quadrillage du carré unitaire (il n'y a pas d'aléa)
        les tensors ont requires_grad=False car cela reviendrait à déformer la grille"""
        points_dim = [np.linspace(-1, 1, self.n_grid_step) for _ in range(self.input_dim)]
        points_dim = np.meshgrid(*points_dim)
        
        def f(*point_dim):
            v = torch.tensor(list(point_dim)).float()
            v.requires_grad = False
            return v
        
        return np.vectorize(f, otypes=[object])(*points_dim).reshape(-1)
    
    def forward(self, x):
        # concatenation de x et p pour tous les points de la grille
        grid_x = [torch.cat((x, p)) for p in self.grid]
        
        # f(x_p) pour tout x_p pour tout mlp
        return self._nuage_tmp.recreate(np.array([[f(x_p) for x_p in grid_x] for f in self.decodeurs]).reshape(-1))


class Reconstructeur2(nn.Module):
    """chamfer quadratique, à supprimer"""
    def __init__(self, n_mlp, latent_size, n_grid_step):
        super().__init__()
        print("nombre de MLP:", n_mlp)
        print("résolution de la grille:", n_grid_step, "*", n_grid_step, "=", n_grid_step ** 2)
        print("taille des nuages générés:", n_mlp, "*", n_grid_step ** 2, "=", n_mlp * n_grid_step ** 2, )
        self.input_dim = 2
        self.output_dim = 3
        self.n_grid_step = n_grid_step
        self.verbose = True
        
        self.decodeurs = nn.ModuleList([nn.Sequential(
            #on rajoute les coordonnées d'un point dans l'espace de départ
            nn.Linear(latent_size + self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            #on retourne un point dans l'espace de sortie
            nn.Linear(128, self.output_dim),
            #tanh ramène dans [-1, 1], comme les nuages de notre auto-encodeur et de groueix
            nn.Tanh()
        ) for _ in range(n_mlp)])
        
        self.decodeurs = nn.ModuleList([nn.Sequential(
            nn.Linear(latent_size + self.input_dim, self.output_dim),
            nn.Tanh()
        ) for _ in range(n_mlp)])
        
        self.grid = self._gen_grid()
        self.loss = self.chamfer
    
    def _gen_grid(self):
        points_dim = [np.linspace(0, 1, self.n_grid_step) for _ in range(self.input_dim)]
        points_dim = np.meshgrid(*points_dim)
        
        def f(*point_dim):
            v = torch.tensor(list(point_dim)).float()
            v.requires_grad = False
            return v
        
        return np.vectorize(f, otypes=[object])(*points_dim).reshape(-1)
    
    @staticmethod
    def chamfer(Y, S):
        """return chamfer loss between
        the generated set Y = {f(x, p) for each f, p}
        and the real pointcloud S corresponding to the latent vector x

        sum_F(sum_A) et min_A(min_F) correspondent à une simple somme ou min sur l'ensemble des données
        donc on représente l'ensemble des points générés Y comme une liste et non comme une matrice
        """
    
        # Pas la même taille :
        # normes = torch.pow(torch.norm(Y - S), 2)
    
        # normes = np.array([[torch.norm(y - s) for s in S] for y in Y])
        normes = np.array([[torch.sum(torch.pow(y - s, 2), dim=(0,)) for s in S] for y in Y])
        loss1 = np.sum(np.min(normes, axis=0))
        loss2 = np.sum(np.min(normes, axis=1))
        return loss1, loss2
    
        # On sert uniquement des min selon les deux axes, on peut ne pas stocker la matrice pour éviter les problèmes de RAM
        # listes des minimums sur chaque ligne et colonne
        min_axe0 = [torch.tensor(float("+inf"))] * len(S)
        min_axe1 = [torch.tensor(float("+inf"))] * len(Y)
    
        # pour chaque case de la matrice, met à jour les deux minimums correspondants
        for i, s in enumerate(S):
            for j, y in enumerate(Y):
                val = torch.norm(y - s)
                min_axe0[i] = torch.min(val, min_axe0[i])
                min_axe1[j] = torch.min(val, min_axe1[j])
    
        # les Q doivent bien être atteint par un MLP
        loss1 = sum(min_axe0)
    
        # les MLP doivent bien atteindre un Q
        loss2 = sum(min_axe1)
    
        return loss1, loss2

    def forward(self, x):
        # concatenation de x et p pour tous les points de la grille
        grid_x = [torch.cat((x, p)) for p in self.grid]
        
        # f(x_p) pour tout x_p pour tout mlp
        return np.array([[f(x_p) for x_p in grid_x] for f in self.decodeurs]).reshape(-1)


def fit_reconstructeur(reconstructeur, x_train, y_train, x_test, y_test, epochs, lr=1e-5, grid_scale=1.0):
    """grid_scale:
        les points 3D générés sont calculés à partir d'un échantillonage d'un carré 1*1,
        prendre un carré 100*100 revient à augmenter les coefficients du premier Linear sans pénaliser le modèle
    """
    reconstructeur.grid *= grid_scale
    optimizer = torch.optim.Adam(reconstructeur.parameters(), lr=lr)
    loss_train = None
    
    time_loss = np.zeros(epochs)
    time_tot = np.zeros(epochs)
    
    for epoch in range(epochs):
        loss_train = 0
        for x, y in zip(x_train, y_train):
            assert not x.requires_grad
            
            t_tot = time.time()
            
            y_pred = reconstructeur.forward(x)
            
            t_loss = time.time()
            loss = reconstructeur.loss(y_pred, y)
            time_loss[epoch] += time.time() - t_loss
            
            # print(loss)
            loss = loss[0] + loss[1]
            loss_train += loss.item() / len(x_train)
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            time_tot[epoch] += time.time() - t_tot
        
        if reconstructeur.verbose and (epochs < 10 or epoch % (epochs // 10) == 0):
            reconstructeur.eval()
            
            s_test = sum(sum(reconstructeur.loss(reconstructeur.forward(x), y.data)).item() for (x, y) in zip(x_test, y_test))
            
            print("time", epoch,
                  "loss train %.1e" % loss_train,
                  "loss test %.1e" % s_test)
            reconstructeur.train()
        
        if loss_train / len(x_train) < 1e-8:
            #convergence finie, ca ne sert à rien de continuer
            break
    
    #apprentissage fini, passage en mode évaluation
    reconstructeur.eval()
    return loss_train, time_loss, time_tot


def draw_cloud(ax, c):
    ax.scatter(c[:, 0], c[:, 1], c[:, 2])


def plot_tailles(clouds):
    import matplotlib.pyplot as plt
    tailles = np.array([[len(x) for x in n.mat.reshape(-1)] for n in clouds], dtype=int).reshape(-1)
    plt.hist(tailles, bins=20, range=(1, max(tailles)))
    plt.show()


def _main():
    chosen_subset = [0]
    n_per_cat = 1
    clouds2 = tSNE.get_clouds(chosen_subset, n_per_cat, ratio=.01)
    print("taille des nuages ground truth:", len(clouds2[0]))
    latent = tSNE.get_latent(chosen_subset, n_per_cat,
                             nPerObj=1)  # /!\ attention: il faut que les fichiers sur le disque correspondent
    
    segmentation = Segmentation(5)
    
    clouds = [Nuage(x, segmentation) for x in clouds2]
    
    # plot_tailles(clouds)
    #tSNE.write_clouds("./data/output_clouds/ground_truth", [[p.detach().numpy() for p in clouds[0]]])
    
    ratio_train = 1
    n_train = int(ratio_train * len(latent))
    
    indexes = list(range(len(latent)))
    #random.shuffle(indexes)
    train_x = [latent[i] for i in indexes[:n_train]]
    test_x = [latent[i] for i in indexes[n_train:]]
    train_y = [clouds[i] for i in indexes[:n_train]]
    test_y = [clouds[i] for i in indexes[n_train:]]
    train_y2 = [clouds2[i] for i in indexes[:n_train]]
    test_y2 = [clouds2[i] for i in indexes[n_train:]]
    
    n_mlp = 1
    latent_size = 25088  # défini par l'encodeur utilisé
    grid_points = 1
    epochs = 10
    grid_size = 1e0
    lr = 1e-1
    
    reconstructeur = Reconstructeur(n_mlp, latent_size, grid_points, segmentation)
    
    t = time.time()
    for _ in range(1):
        loss, t_loss, t_tot = fit_reconstructeur(reconstructeur, train_x, train_y, test_x, test_y, epochs, lr=lr, grid_scale=grid_size)
        print("temps: ")
        print("loss", sum(t_loss), t_loss)
        print("tot", sum(t_tot), t_tot)
        print("ratio", sum(t_loss)/sum(t_tot), t_loss/t_tot)
    print("O(n)", time.time() - t)
    
    print("répartition point segmentation")
    print(np.histogram(Nuage.points_parcourus))
    print(np.mean(Nuage.points_parcourus))
    # print(segmentation.distances_2)
    
    output = reconstructeur.forward(latent[0])
    output = [p.detach().numpy() for p in output.liste_points]
    tSNE.write_clouds("./data/output_clouds", [output])
    
    print("reconstructeur O(n2) : ")
    reconstructeur2 = Reconstructeur2(n_mlp, latent_size, grid_points)
    t = time.time()
    for _ in range(1):
        fit_reconstructeur(reconstructeur2, train_x, train_y2, test_x, test_y2, epochs, lr=lr, grid_scale=grid_size)
    print("O(n2)", time.time() - t)  #


if __name__ == '__main__':
    _main()


# torch.save(the_model.state_dict(), PATH)
# the_model = TheModelClass(*args, **kwargs)
# the_model.load_state_dict(torch.load(PATH))
