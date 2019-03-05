import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import random
import tSNE


chamfer_verbose = True
def chamfer(Y, S):
    """return chamfer loss between
    the generated set Y = {f(x, p) for each f, p}
    and the real set S corresponding to x
    
    sum_F(sum_A) et min_A(min_F) correspondent à sum() et min() sur l'ensemble des nombres de la matrice
    donc on représente l'ensemble des points générés Y par une liste et non une matrice
    """
    global chamfer_verbose

    if chamfer_verbose:
        print("Y:", type(Y), Y.shape)
        print("S:", type(S), S.shape)
    
    # Je n'arrive pas à travailler avec les tensors directement :
    # Pas la même taille :
    # normes = torch.pow(torch.norm(Y - S), 2)
    # Breaks the computational graph :
    # normes = torch.tensor([... for s, y])
    # loss = torch.sum(torch.min(normes, dim=0)[0]) #min() renvoie (min, argmin)
    
    # On sert uniquement des min selon les deux axes, on peut ne pas stocker la matrice pour éviter les problèmes de RAM
    # Sinon on ferait :
    # normes = np.array([[torch.pow(torch.norm(y - s), 2) for s in S] for y in Y])
    # loss1 = np.sum(np.min(normes, axis=1), loss2 = np.sum(np.min(normes, axis=0))
    
    min_axe0 = [float("+inf")] * len(S)
    min_axe1 = [float("+inf")] * len(Y)
    for i, s in enumerate(S):
        for j, y in enumerate(Y):
            val = torch.pow(torch.norm(y - s), 2)
            min_axe0[i] = min(min_axe0[i], val)
            min_axe1[j] = min(min_axe1[j], val)
    
    # les Q doivent bien être atteint par un MLP
    loss1 = sum(min_axe0)
    
    # les MLP doivent bien atteindre un Q
    loss2 = sum(min_axe1)
    
    if chamfer_verbose:
        #print(type(normes), normes.size())
        #print(loss1.item(), loss2.item())
        #print(type(normes), normes.shape)
        print("loss:", type(loss1), loss1.item(), loss2.item())
        chamfer_verbose = False
    
    return loss1 + loss2
    

class Reconstructeur(nn.Module):
    def __init__(self, n_mlp, space_dim, latent_size, grid_points, grid_size, epochs):
        super().__init__()
        self.space_dim = space_dim
        self.epochs = epochs
        
        self.decodeurs = nn.ModuleList([nn.Sequential(
            nn.Linear(latent_size + space_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, space_dim),
            nn.Tanh()
        ) for _ in range(n_mlp)])
        
        points_dim = [np.linspace(0, grid_size, grid_points) for _ in range(space_dim)]
        points_dim = np.meshgrid(*points_dim)
        
        def f(*point_dim):
            v = Variable(torch.tensor(list(point_dim)).float())
            v.require_grad = False
            return v
        
        #liste de n points 3D
        self.grid = np.vectorize(f, otypes=[object])(*points_dim).reshape(-1)
        self.verbose = True
    
    def forward(self, x):
        grid_x = [torch.cat((x, p)) for p in self.grid]
        return np.array([[f(x_p) for x_p in grid_x] for f in self.decodeurs]).reshape(-1)
        
        # res = torch.tensor([[f(x_p).detach().numpy() for x_p in grid_x] for f in self.decodeurs])
        # res.requires_grad = True
        # return res
        
    
    def fit(self, x_train, y_train, x_test, y_test):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    
        for t in range(self.epochs):
            s_train = 0
            for x, y in zip(x_train, y_train):
                assert not x.requires_grad
                
                y_pred = self.forward(x)
                #assert y_pred.requires_grad
                
                loss = chamfer(y_pred, y)
                s_train += loss.item() / len(x_train)
                
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            if self.verbose and (self.epochs < 10 or t % (self.epochs // 10) == 0):
                self.eval()
            
                s_test = sum(chamfer(self.forward(x), y.data).item() for (x, y) in zip(x_test, y_test))
            
                print("time", t,
                      "loss train %.1e" % s_train,
                      "loss test %.1e" % s_test)
                self.train()
            
                if s_train / len(x_train) < 1e-8:
                    #convergence finie, ca ne sert à rien de continuer
                    break
    
        #apprentissage fini, passage en mode évaluation
        self.eval()
        return s_train


def draw_cloud(ax, c):
    ax.scatter(c[:, 0], c[:, 1], c[:, 2])
    
def _main():
    chosen_subset = [0]
    n_per_cat = 1
    latent = tSNE.get_latent(chosen_subset, n_per_cat, nPerObj=1)
    clouds = tSNE.get_clouds(chosen_subset, n_per_cat, ratio=.002)

    tSNE.write_clouds("./data/output_clouds/ground_truth", [[p.detach().numpy() for p in clouds[0]]])
    exit()
    ratio_train = 1
    n_train = int(ratio_train * len(latent))
    
    indexes = list(range(len(latent)))
    #random.shuffle(indexes)
    train_x = [latent[i] for i in indexes[:n_train]]
    test_x = [latent[i] for i in indexes[n_train:]]
    train_y = [clouds[i] for i in indexes[:n_train]]
    test_y = [clouds[i] for i in indexes[n_train:]]

    n_mlp = 10
    space_dim = 3
    latent_size = 25088 # défini par l'encodeur utilisé
    grid_points = 2 # au cube
    grid_size = 100
    epochs = 10
    
    reconstructeur = Reconstructeur(n_mlp, space_dim, latent_size, grid_points, grid_size, epochs)
    
    reconstructeur.fit(train_x, train_y, test_x, test_y)
    
    output = reconstructeur.forward(latent[0])
    output = [p.detach().numpy() for p in output]
    tSNE.write_clouds("./data/output_clouds", [output])

if __name__ == '__main__':
    _main()
    
