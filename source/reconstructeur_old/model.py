import torch
from torch import nn
import numpy as np
import time
from nuage import Nuage


class Reconstructeur(nn.Module):
    def __init__(self, n_mlp, latent_size, n_grid_step, segmentation, quadratic=False):
        super().__init__()
        self.nuage_tmp = Nuage([], segmentation)
        self.input_dim = 2
        self.output_dim = 3
        self.n_grid_step = n_grid_step
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
        # noinspection PyTypeChecker
        y0_a = torch.cat([f(x0) for f in self.f0_a])
        y0_a = y0_a.reshape((self.n_mlp, 1, -1))
        
        # calcul pour tout f pour tout p
        # noinspection PyTypeChecker
        y0_b = torch.cat([torch.cat([f(p) for p in self.grid]) for f in self.f0_b])
        y0_b = y0_b.reshape((self.n_mlp, len(self.grid), -1))
        
        # somme les deux parties
        y0 = torch.add(y0_a, y0_b)
        
        # applique le reste du réseau
        # noinspection PyTypeChecker
        res = torch.cat([torch.cat([f(x) for x in e]) for f, e in zip(self.decodeurs, y0)])
        res = res.reshape((self.n_mlp * len(self.grid), self.output_dim))
        return res
    

def fit_reconstructeur(reconstructeur, train, epochs, sample_size=None,
                       lr=1e-4, lr_decay=.05, grid_scale=1, loss_factor=1.0,
                       ind_cloud_saved=range(0), test=None, list_epoch_loss=range(0)):
    """
    grid_scale:
        les points 3D générés sont calculés à partir d'un échantillonage d'un carré 1*1,
        prendre un carré 100*100 revient à augmenter les coefficients du premier Linear sans pénaliser le modèle
        
    ind_predicted:
        indices dans train ou test pour lesquels sauvegarder l'évolution des prédictions
        (les indices de test suivent ceux de train)
    
    ind_plotted:
        indices pour lesquels calculer la loss en test
    """
    if type(loss_factor) == float:
        loss_factor = [loss_factor]*epochs
    assert len(loss_factor) == epochs
    
    if sample_size is None:
        # considère tous les points à chaque itération
        sample_size = len(train[1][0].liste_points)
    
    list_predicted = {i:[] for i in ind_cloud_saved}
    list_loss_train = []
    list_loss_test = []
    time_tot = time.time()
    time_loss = 0
    
    reconstructeur.grid *= grid_scale
    optimizer = torch.optim.Adagrad(reconstructeur.parameters(), lr=lr, lr_decay=lr_decay)
    # optimizer = torch.optim.SGD(reconstructeur.parameters(), lr=lr)
    for epoch in range(epochs):
        loss_train = 0
        loss_test = 0
        
        with torch.no_grad():
            reconstructeur.eval()
            # calcule la loss en test
            if epoch in list_epoch_loss:
                # ne se sert pas du même loss_factor que pour train
                if test is not None and len(test[0]):
                    loss_test = [reconstructeur.loss(reconstructeur.nuage_tmp.recreate(reconstructeur.forward(x)),
                                                     y.sub_sample(sample_size), k=1)
                              for (x, y) in zip(test[0], test[1])]
                    loss_test = sum([s[0].item() + s[1].item() for s in loss_test])
                    list_loss_test.append(loss_test/len(test[1]))
            
            # sauvegarde pour l'animation
            for i in ind_cloud_saved:
                # les tests sont considérés comme étant juste après les train
                if i-len(train[0]) >= 0:
                    list_predicted[i].append(reconstructeur.forward(test[0][i-len(train[0])]).detach().numpy())
                
            reconstructeur.train()
        
        # train
        for i in range(len(train[0])):
            x = train[0][i]
            y = train[1][i].sub_sample(sample_size)
            assert not x.requires_grad
            
            y_pred = reconstructeur.forward(x)
            
            if i in ind_cloud_saved:
                list_predicted[i].append(y_pred.detach().numpy().copy())
            
            time0 = time.time()
            y_pred = reconstructeur.nuage_tmp.recreate(y_pred)
            
            loss = reconstructeur.loss(y_pred, y, loss_factor[epoch])
            time_loss += time.time() - time0
            
            print("loss", loss)
            loss = loss[0] + loss[1]
            loss_train += loss.item()
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        list_loss_train.append(loss_train / len(train[1]))
            
        if epoch in list_epoch_loss:
            print("time", epoch,
                  "loss train %.3e" % loss_train,
                  "loss test %.3e" % loss_test)
        
    #apprentissage fini, passage en mode évaluation
    reconstructeur.eval()
    time_tot = time.time() - time_tot
    print("temps loss", time_loss," tot", time_tot, "ratio", time_loss/time_tot)
    return {"loss_train": list_loss_train,
            "loss_test":  list_loss_test,
            "predicted":  list_predicted,
            "time":       (time_tot, time_loss, time_loss/time_tot),
            }


def _test():
    from segmentation import Segmentation
    import input_output

    s = Segmentation(5)
    c1 = Nuage(input_output.get_clouds([0], 1, size=.01)[0], s)
    c2 = torch.tensor([[0., 0, 0], [1, 1, 1], [-1, -1, -1], [.2, .3, .0]])
    c2 = Nuage(c2, s)
    n_mlp = 2; latent_size = 25088; grid_points = 4; epochs = 5
    latent = input_output.get_latent([0], 1, nPerObj=1)[0]
    
    # équivalence des deux fonctions de loss (avec points écrits à la main)
    print("seg",  Nuage.chamfer_seg(c1, c2))
    print("quad", Nuage.chamfer_quad(c1.liste_points, c2.liste_points))
    assert Nuage.chamfer_seg(c1, c2) == Nuage.chamfer_quad(c1.liste_points, c2.liste_points)
    
    # presque équivalence de convergence
    # (utilise les mêmes nombres aléatoires pour avoir le même résultat)
    torch.manual_seed(0);np.random.seed(0)
    reconstructeur1 = Reconstructeur(n_mlp, latent_size, grid_points, s, quadratic=False)
    torch.manual_seed(0);np.random.seed(0)
    reconstructeur2 = Reconstructeur(n_mlp, latent_size, grid_points, s, quadratic=True)
    
    res1 = fit_reconstructeur(reconstructeur1, ([latent], [c1]), epochs)
    print()
    res2 = fit_reconstructeur(reconstructeur2, ([latent], [c1]), epochs)
    
    # différences sûrement dues à des erreurs d'arrondi
    assert np.all(np.abs(np.array(res1["loss_train"]) - np.array(res2["loss_train"])) < 1e-3)
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
