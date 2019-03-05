import torch
from torch import nn
import numpy as np
import tSNE


class Segmentation:
    def __init__(self, n_step):
        self.dim = 3
        self.coo_min = -1
        self.coo_max = 1
        self.n_step = n_step
    
    def gen_matrix(self):
        return np.array([[[]for _ in range(self.n_step)] for _ in range(self.n_step)])
    
    def get_mat_coo(self, point):
        return [(coo - self.coo_min) / self.n_step for coo in point]
    
    def get_closest_neigbhour(self, point):
        return []
    
class Nuage:
    """stocke l'ensemble des points sous forme d'une liste de points et d'un tableau à 3 dimensions"""
    def __init__(self, points, segmentation):
        self.segmentation = segmentation
        self.liste_points = points
        self.mat = segmentation.gen_matrix()
        
        for p in self.liste_points:
            self.mat[self.segmentation.get_mat_coo(p)].append(p)
    
    def recreate(self, points):
        """vide et re-rempli l'ancien tableau plutôt que de le supprimer et d'en allouer un nouveau"""
        for p in self.liste_points:
            self.mat[self.segmentation.get_mat_coo(p)]=[]
        
        self.liste_points = points
        
        for p in self.liste_points:
            self.mat[self.segmentation.get_mat_coo(p)].append(p)
    
    def get_closest(self, point):
    
    
        
        
def chamfer(Y, S):
    """return chamfer loss between
    the generated set Y = {f(x, p) for each f, p}
    and the real pointcloud S corresponding to the latent vector x

    sum_F(sum_A) et min_A(min_F) correspondent à une simple somme ou min sur l'ensemble des données
    donc on représente l'ensemble des points générés Y comme une liste et non comme une matrice
    """
    
    # Pas la même taille :
    # normes = torch.pow(torch.norm(Y - S), 2)
    
    # On sert uniquement des min selon les deux axes, on peut ne pas stocker la matrice pour éviter les problèmes de RAM
    # Sinon on ferait :
    # normes = np.array([[torch.pow(torch.norm(y - s), 2) for s in S] for y in Y])
    # loss1 = np.sum(np.min(normes, axis=1), loss2 = np.sum(np.min(normes, axis=0))
    
    # listes des minimums sur chaque ligne et colonne
    min_axe0 = [torch.tensor(float("+inf"))] * len(S)
    min_axe1 = [torch.tensor(float("+inf"))] * len(Y)
    
    # pour chaque case de la matrice, met à jour les deux minimums correspondants
    for i, s in enumerate(S):
        for j, y in enumerate(Y):
            val = torch.pow(torch.norm(y - s), 2)
            min_axe0[i] = torch.min(val, min_axe0[i])
            min_axe1[j] = torch.min(val, min_axe1[j])
    
    # les Q doivent bien être atteint par un MLP
    loss1 = sum(min_axe0)
    
    # les MLP doivent bien atteindre un Q
    loss2 = sum(min_axe1)
    
    return loss1, loss2


class Reconstructeur(nn.Module):
    def __init__(self, n_mlp, latent_size, n_grid_step):
        super().__init__()
        print("nombre de MLP:", n_mlp)
        print("résolution de la grille:", n_grid_step, "*", n_grid_step, "=", n_grid_step**2)
        print("taille des nuages générés:", n_mlp * n_grid_step**2)
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
    
    def _gen_grid(self):
        """retourne une liste de n points 2D au bon format :
        array(tensor(coo_x, coo_y))
        les points forment un quadrillage du carré unitaire (il n'y a pas d'aléa)
        les tensors ont requires_grad=False car cela reviendrait à déformer la grille"""
        points_dim = [np.linspace(0, 1, self.n_grid_step) for _ in range(self.input_dim)]
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
        return np.array([[f(x_p) for x_p in grid_x] for f in self.decodeurs]).reshape(-1)
    
def fit_reconstructeur(reconstructeur, x_train, y_train, x_test, y_test, epochs, grid_scale=1.0):
    """grid_scale:
        les points 3D générés sont calculés à partir d'un échantillonage d'un carré 1*1,
        prendre un carré 100*100 revient à augmenter les coefficients du premier Linear sans pénaliser le modèle
    """
    reconstructeur.grid *= grid_scale
    optimizer = torch.optim.Adam(reconstructeur.parameters(), lr=1e-4)
    loss_train = None

    for t in range(epochs):
        loss_train = 0
        for x, y in zip(x_train, y_train):
            assert not x.requires_grad
            
            y_pred = reconstructeur.forward(x)
            #assert y_pred.requires_grad
            
            loss = chamfer(y_pred, y)
            loss = loss[0] + loss[1]
            loss_train += loss.item() / len(x_train)
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if reconstructeur.verbose and (epochs < 10 or t % (epochs // 10) == 0):
            reconstructeur.eval()
            
            s_test = sum(sum(chamfer(reconstructeur.forward(x), y.data)).item() for (x, y) in zip(x_test, y_test))
            
            print("time", t,
                  "loss train %.1e" % loss_train,
                  "loss test %.1e" % s_test)
            reconstructeur.train()
            
            if loss_train / len(x_train) < 1e-8:
                #convergence finie, ca ne sert à rien de continuer
                break
    
    #apprentissage fini, passage en mode évaluation
    reconstructeur.eval()
    return loss_train


def draw_cloud(ax, c):
    ax.scatter(c[:, 0], c[:, 1], c[:, 2])


def _main():
    chosen_subset = [0]
    n_per_cat = 1
    clouds, classes, identifiants = tSNE.get_clouds(chosen_subset, n_per_cat, ratio=.002)
    latent = tSNE.get_corresponding_latent(classes, identifiants, n_per_cat)
    
    #tSNE.write_clouds("./data/output_clouds/ground_truth", [[p.detach().numpy() for p in clouds[0]]])
    
    ratio_train = 1
    n_train = int(ratio_train * len(latent))
    
    indexes = list(range(len(latent)))
    #random.shuffle(indexes)
    train_x = [latent[i] for i in indexes[:n_train]]
    test_x = [latent[i] for i in indexes[n_train:]]
    train_y = [clouds[i] for i in indexes[:n_train]]
    test_y = [clouds[i] for i in indexes[n_train:]]
    
    n_mlp = 10
    latent_size = 25088  # défini par l'encodeur utilisé
    grid_points = 2
    epochs = 10
    grid_size = 100
    
    reconstructeur = Reconstructeur(n_mlp, latent_size, grid_points)
    
    fit_reconstructeur(reconstructeur, train_x, train_y, test_x, test_y, epochs, grid_size)
    
    output = reconstructeur.forward(latent[0])
    output = [p.detach().numpy() for p in output]
    tSNE.write_clouds("./data/output_clouds", [output])


if __name__ == '__main__':
    _main()

"""
moins de points ex 30
espace latent de 20
rajouter une couche 64

stats par type de forme err std

visualiser regroupement en clusters avec espace latent de taille 2

google colab"""
