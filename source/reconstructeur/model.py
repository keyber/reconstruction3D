import torch
from torch import nn
import numpy as np
import time
from kdtree import Nuage


class Reconstructeur(nn.Module):
    def __init__(self, n_mlp, latent_size, n_grid_step, loss):
        super().__init__()
        self.input_dim = 2
        self.output_dim = 3
        self.n_grid_step = n_grid_step
        # la distance entre deux points vaut [xmax=1 - xmin=-1]=2 / (n_grid_step - 1)
        # on peut rajouter un nombre aléatoire inférieur à la moitié
        self.grid_rand = 1 / (n_grid_step - 1)
        self.n_mlp = n_mlp
        self.loss = loss
        
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

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                # m.bias.data.fill_(0.01)
                
        for net in self.decodeurs:
            net.apply(init_weights)
        
        self.grid = self._gen_grid()
        
    def _gen_grid(self):
        """retourne une liste de n points 2D sous la forme d'un tensor (n, 2)
        les points forment un quadrillage du carré unitaire (il n'y a pas d'aléa)
        le tensor a requires_grad=False car cela reviendrait à déformer la grille"""
        points_dim = [np.linspace(-1, 1, self.n_grid_step) for _ in range(self.input_dim)]
        x_coo, y_coo = np.meshgrid(*points_dim)
        x_coo, y_coo = x_coo.reshape(-1), y_coo.reshape(-1)
        grid = np.array([[x,y] for (x,y) in zip(x_coo, y_coo)])
        grid = torch.tensor(grid).float()
        grid.requires_grad = False
        return grid

    # noinspection PyTypeChecker
    def forward(self, x0):
        # partie indépendante du sampling :
        y0_a = torch.cat([f(x0) for f in self.f0_a])
        y0_a = y0_a.reshape((self.n_mlp, 1, -1))
        
        # calcul pour tout f pour tout p
        grid = self.grid + torch.rand(self.grid.shape) * self.grid_rand
        
        y0_b = torch.cat([torch.cat([f(p) for p in grid]) for f in self.f0_b])
        y0_b = y0_b.reshape((self.n_mlp, len(self.grid), -1))
        
        # somme les deux parties
        y0 = torch.add(y0_a, y0_b)
        
        # applique le reste du réseau
        res = torch.cat([torch.cat([f(x) for x in e]) for f, e in zip(self.decodeurs, y0)])
        res = res.reshape((self.n_mlp * len(self.grid), self.output_dim))
        return res
    

#todo refactor
def fit_reconstructeur(reconstructeur, train, epochs, sample_size=None, mini_batch_size=1,
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
        sub_sampling = [None] * epochs
    else:
        ones = torch.ones(len(train[1][0].points))
        sub_sampling = [torch.multinomial(ones, sample_size) for _ in range(epochs)]
    
    list_predicted = {i:[] for i in ind_cloud_saved}
    list_loss_train = []
    list_loss_test = []
    list_loss_detailled = [[] for _ in range(len(train[0]))]
    time_tot = time.time()
    time_loss = 0
    
    reconstructeur.grid *= grid_scale
    optimizer = torch.optim.Adagrad(reconstructeur.parameters(), lr=lr, lr_decay=lr_decay)
    for epoch in range(epochs):
        loss_train = [0,0]
        
        with torch.no_grad():
            reconstructeur.eval()
            # calcule la loss en test
            if epoch in list_epoch_loss:
                # ne se sert pas du même loss_factor que pour train
                if test is not None and len(test[0]):
                    loss_test = [reconstructeur.loss(Nuage(reconstructeur.forward(x), eps=0), y, sub_sampling=sub_sampling[epoch], k=1)
                              for (x, y) in zip(test[0], test[1])]
                    s0 = sum([s[0].item() for s in loss_test]) / len(test[1])
                    s1 = sum([s[1].item() for s in loss_test]) / len(test[1])
                    list_loss_test.append((s0, s1))
            
            # sauvegarde pour l'animation
            for i in ind_cloud_saved:
                # les tests sont considérés comme étant juste après les train
                if i-len(train[0]) >= 0:
                    list_predicted[i].append(reconstructeur.forward(test[0][i-len(train[0])]).detach().numpy())
                
            reconstructeur.train()
        
        # train
        for i in range(0, len(train[0]), mini_batch_size):
            mini_batch_loss = None
            for j in range(i, min(i+mini_batch_size, len(train[0]))):
                x = train[0][j]
                y = train[1][j]
                assert not x.requires_grad
                
                y_pred = reconstructeur.forward(x)
                
                if j in ind_cloud_saved:
                    list_predicted[j].append(y_pred.detach().numpy().copy())
                
                time0 = time.time()
                y_pred = Nuage(y_pred, eps=0)
                loss = reconstructeur.loss(y_pred, y, k=loss_factor[epoch], sub_sampling=sub_sampling[epoch])
                time_loss += time.time() - time0
                
                print("loss", loss)
                loss_train[0] += loss[0].item()
                loss_train[1] += loss[1].item()
                
                list_loss_detailled[j].append((loss[0].item(), loss[1].item()))
                
                loss = loss[0] + loss[1]
                mini_batch_loss = loss if mini_batch_loss is None else mini_batch_loss + loss
                
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            mini_batch_loss.backward()
            optimizer.step()

        list_loss_train.append((loss_train[0] / len(train[1]), loss_train[1] / len(train[1])))
        
        if epoch in list_epoch_loss:
            print("time", epoch,
                  "loss train %.3e" % (list_loss_train[-1][0]+list_loss_train[-1][1]),
                  "loss test %.3e" % (list_loss_test[-1][0]+list_loss_test[-1][1] if len(test[0]) else 0))
        
    #apprentissage fini, passage en mode évaluation
    reconstructeur.eval()
    time_tot = time.time() - time_tot
    print("temps loss", time_loss," tot", time_tot, "ratio", time_loss/time_tot)
    return {"loss_train": list_loss_train,
            "loss_test":  list_loss_test,
            "loss_detailled":  list_loss_detailled,
            "predicted":  list_predicted,
            "time":       (time_tot, time_loss, time_loss/time_tot),
            }


def _test():
    c1 = Nuage(input_output.get_clouds([0], 1, size=.01)[0], eps=0)
    c2 = torch.tensor([[0., 0, 0], [1, 1, 1], [-1, -1, -1], [.2, .3, .0]])
    c2 = Nuage(c2, eps=0)
    
    # équivalence de la loss naïve et avec kdtree
    l0 = c1.chamfer(c2)
    l1 = c2.chamfer(c1)
    l1 = l1[1], l1[0]
    l2 = c1.chamfer_quadratic(c2)
    l3 = c2.chamfer_quadratic(c1)
    l3 = l3[1], l3[0]
    assert l0 == l1
    assert l2 == l3
    assert l0 == l2
    
    n_mlp = 2; latent_size = 25088; grid_points = 4; epochs = 5
    latent = input_output.get_latent([0], 1, nPerObj=1)[0]
    # presque équivalence de convergence
    # (utilise les mêmes nombres aléatoires pour avoir le même résultat)
    torch.manual_seed(0);np.random.seed(0)
    reconstructeur1 = Reconstructeur(n_mlp, latent_size, grid_points, Nuage.chamfer)
    torch.manual_seed(0);np.random.seed(0)
    reconstructeur2 = Reconstructeur(n_mlp, latent_size, grid_points, Nuage.chamfer_quadratic)
    
    torch.manual_seed(0);np.random.seed(0)
    res1 = fit_reconstructeur(reconstructeur1, ([latent], [c1]), epochs)
    torch.manual_seed(0);np.random.seed(0)
    res2 = fit_reconstructeur(reconstructeur2, ([latent], [c1]), epochs)
    
    # différences sûrement dues à des erreurs d'arrondi
    m = np.max(np.abs(np.array(res1["loss_train"]) - np.array(res2["loss_train"])))
    print("diff loss", m)
    assert m < 1e-6


def _test2():
    c1 = torch.tensor([[0., 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]])
    c1 = Nuage(c1, eps=0)
    
    assert 2*sum(c1.chamfer(c1)) == 0

    for _ in range(10):
        a = np.random.random() * 2 - 1
        c2 = Nuage(torch.tensor([[a, 0, 0]]), eps=0)
        l = c2.chamfer(c1)
        assert abs(2 * 4 * l[0] - (1 + 1 + 2 + 4 * a * a)) < 1e-6
        assert abs(2 * 1 * l[1] - a * a) < 1e-6


if __name__ == '__main__':
    import input_output
    _test()
    _test2()
    print("tests passés")

# torch.save(the_model.state_dict(), PATH)
# the_model = TheModelClass(*args, **kwargs)
# the_model.load_state_dict(torch.load(PATH))
