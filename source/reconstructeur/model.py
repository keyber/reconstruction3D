import torch
from torch import nn
import numpy as np
import time
from nuage import Nuage


class Reconstructeur(nn.Module):
    def __init__(self, n_mlp, latent_size, n_grid_step, segmentation, quadratic):
        super().__init__()
        self._nuage_tmp = Nuage([], segmentation)
        self.input_dim = 2
        self.output_dim = 3
        self.n_grid_step = n_grid_step
        self.verbose = True
        self.n_mlp = n_mlp
        
        # s = [128, 64]
        s = [512, 256, 128]
        self.sizes = s
        
        # cf forward
        self.f0_a = nn.ModuleList([nn.Linear(latent_size, s[0]) for _ in range(n_mlp)])
        self.f0_b = nn.ModuleList([nn.Linear(self.input_dim, s[0]) for _ in range(n_mlp)])
        
        
        self.decodeurs = nn.ModuleList([nn.Sequential(
            nn.ReLU(),
            
            nn.Linear(s[0], s[1]),
            nn.ReLU(),
            
            nn.Linear(s[1], s[2]),
            nn.ReLU(),
            
            #on retourne un point dans l'espace de sortie
            nn.Linear(s[-1], self.output_dim),
            #tanh ramène dans [-1, 1], comme les nuages de notre auto-encodeur et de groueix
            nn.Tanh()
        ) for _ in range(n_mlp)])
        
        self.grid = self._gen_grid()
        
        self.loss = Nuage.chamfer_quad if quadratic else Nuage.chamfer_seg
    
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
    
    def forward(self, x0):
        # partie indépendante du sampling :
        y0_a = torch.cat([f(x0) for f in self.f0_a])
        y0_a = y0_a.reshape((self.n_mlp, 1, -1))
        
        # calcul pour tout f pour tout p
        y0_b = torch.cat([torch.cat([f(p) for p in self.grid]) for f in self.f0_b])
        y0_b = y0_b.reshape((self.n_mlp, len(self.grid), -1))
        
        # combine somme
        y0 = torch.add(y0_a, y0_b)
        
        # le reste
        res = torch.cat([torch.cat([f(x) for x in e]) for f, e in zip(self.decodeurs, y0)])
        res = res.reshape((self.n_mlp * len(self.grid), self.output_dim))
        if self.loss == Nuage.chamfer_seg:
            res = self._nuage_tmp.recreate(res)
        return res
    

def fit_reconstructeur(reconstructeur, x_train, y_train, x_test, y_test, epochs,
                       lr=1e-5, grid_scale=1.0, list_pred=None, loss_factor=1, squared_distances=True):
    """grid_scale:
        les points 3D générés sont calculés à partir d'un échantillonage d'un carré 1*1,
        prendre un carré 100*100 revient à augmenter les coefficients du premier Linear sans pénaliser le modèle
    """
    reconstructeur.grid *= grid_scale
    optimizer = torch.optim.Adagrad(reconstructeur.parameters(), lr=lr, lr_decay=0.05)
    # optimizer = torch.optim.SGD(reconstructeur.parameters(), lr=lr)
    
    time_loss = np.zeros(epochs)
    time_tot = []
    list_loss = []
    
    for epoch in range(epochs):
        loss_train = 0
        t_tot = time.time()
        for x, y in zip(x_train, y_train):
            assert not x.requires_grad
            y_pred = reconstructeur.forward(x)
            
            if list_pred is not None:
                list_pred.append(y_pred.liste_points.detach().numpy().copy())
            
            t_loss = time.time()
            loss = reconstructeur.loss(y_pred, y, loss_factor, squared_distances)
            time_loss[epoch] += time.time() - t_loss
            
            print("loss", loss)
            loss = loss[0] + loss[1]
            loss_train += loss.item() / len(x_train)
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        time_tot.append(time.time() - t_tot)
        list_loss.append(loss_train)
        
        if reconstructeur.verbose and (epochs < 10 or epoch % (epochs // 10) == 0):
            reconstructeur.eval()
            
            s_test = sum(
                sum(reconstructeur.loss(reconstructeur.forward(x), y.data)).item() for (x, y) in zip(x_test, y_test))
            
            print("time", epoch,
                  "loss train %.4e" % loss_train,
                  "loss test %.2e" % s_test)
            reconstructeur.train()
        
        if loss_train / len(x_train) < 1e-8:
            #convergence finie, ca ne sert à rien de continuer
            break
    
    #apprentissage fini, passage en mode évaluation
    reconstructeur.eval()
    return list_loss, time_loss, time_tot


def _test():
    from segmentation import Segmentation
    import input_output

    s = Segmentation(5)
    c1 = Nuage(input_output.get_clouds([0], 1, ratio=.01)[0], s)
    c2 = torch.tensor([[0., 0, 0], [1, 1, 1], [-1, -1, -1], [.2, .3, .0]])
    c2 = Nuage(c2, s)
    n_mlp = 2; latent_size = 25088; grid_points = 4; epochs = 5; lr = 1e-4
    latent = input_output.get_latent([0], 1, nPerObj=1)[0]
    
    # équivalence des deux fonctions de loss (avec points écrits à la main)
    assert Nuage.chamfer_seg(c1, c2) == Nuage.chamfer_quad(c1.liste_points, c2.liste_points)
    
    # presque équivalence de convergence
    # (utilise les mêmes nombres aléatoires pour avoir le même résultat)
    torch.manual_seed(0); np.random.seed(0)
    reconstructeur1 = Reconstructeur(n_mlp, latent_size, grid_points, s, quadratic=False)
    torch.manual_seed(0); np.random.seed(0)
    reconstructeur2 = Reconstructeur(n_mlp, latent_size, grid_points, s, quadratic=True)
    
    loss1, t_loss1, t_tot1 = fit_reconstructeur(reconstructeur1, [latent], [c1], [], [], epochs, lr=lr)
    loss2, t_loss2, t_tot2 = fit_reconstructeur(reconstructeur2, [latent], [c1.liste_points], [], [], epochs, lr=lr)
    
    # différences sûrement dues à des erreurs d'arrondi
    assert np.all(np.abs(np.array(loss1) - np.array(loss2)) < 1e-1)
    print("tests passés")


if __name__ == '__main__':
    _test()

# torch.save(the_model.state_dict(), PATH)
# the_model = TheModelClass(*args, **kwargs)
# the_model.load_state_dict(torch.load(PATH))


"""todo list

calculer/mesurer coût en fonction de
  ground_truth_size
  n_mlp
  grid_point
  (epochs, n_clouds)
  loss chamfer ou simplifiée
  loss distances au carrées ou non


save 3D"""
