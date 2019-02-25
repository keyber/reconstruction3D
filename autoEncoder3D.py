import torch
import numpy as np
import random
from torch.autograd import Variable


def chamfer(Y, S):
    """return chamfer loss between
    the generated set G = {y[p] = f(x,p) for each p}
    and the real set S"""
    #les MLP doivent bien atteindre un Q
    #loss = torch.sum(torch.tensor([torch.min(torch.tensor([torch.pow(torch.norm(y - s), 2).item() for s in S])).item() for y in Y]))
    #les Q doivent bien être atteint par un MLP
    #loss+= torch.sum(torch.tensor([torch.min(torch.tensor([torch.pow(torch.norm(y - s), 2).item() for y in Y])).item() for s in S]))
    
    loss = torch.sum(torch.min(torch.pow(torch.norm(Y - S), 2)))
    loss+= torch.sum(torch.min(torch.pow(torch.norm(Y - S), 2)))
    #for p in P:
    #    loss += min((y[p] - s) ** 2 for s in S)
    #for s in S:
    #    loss += min((y[p] - s) ** 2 for p in P)
    return loss


class AutoEncoder3D(torch.nn.Module):
    def __init__(self, D_in, space_dim, bottleneck_size):
        super().__init__()
        self.space_dim = space_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(D_in * space_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, bottleneck_size),
            torch.nn.ReLU()
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, D_in * space_dim),
            torch.nn.Tanh()
        )
        
    def forward(self, x):
        x = x.view(-1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, self.space_dim)
    
    def fit(self, list_x, epochs, t_eval, test_x):
        """liste de nuages de points 3D"""
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        for t in range(epochs):
            s_train = 0
            for x in list_x:
                assert not x.requires_grad
                
                y_pred = self.forward(x)
                assert y_pred.requires_grad
                
                loss = chamfer(y_pred, x)
                s_train += loss.item()
                
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if t % t_eval == 0:
                self.eval()
                
                s_test = sum(chamfer(self.forward(x), x.data).item() for x in test_x)
                
                print("time", t,
                      "loss train %.0e" % s_train,
                      "loss test %.0e" % s_test)
                self.train()
                
                if s_train / len(list_x) < 1e-8:
                    #convergence finie, ca ne sert à rien de continuer
                    break
        
        #apprentissage fini, passage en mode évaluation
        self.eval()


def gen_sphere(n_spheres, n_points, n_dim=3, center_range=.5, radius_range=.25, bruit=None):
    assert radius_range + center_range <= 1
    assert n_points % n_spheres == 0
    n_per_spheres = n_points // n_spheres
    
    points = []
    for _ in range(n_spheres):
        center = np.random.uniform(-center_range, center_range, n_dim)
        radius = random.uniform(0, radius_range)
        
        for _ in range(n_per_spheres):
            r = radius + (bruit() if bruit is not None else 0)
            orientation = np.random.uniform(-1, 1, n_dim)
            orientation *= r / np.sqrt(np.square(orientation).sum())
            points.append(center + orientation)
    
    return torch.tensor(points).float()
    
    #boules
    #gaussiennes
    #plan #droite #courbes


def gen_plan(n_plan, n_per_plan, bruit=None):
    """plan : ax + by + cz + d = 0
    génère le plan en choisissant aléatoirement a, b et d (mais pas c)
    génère les coordonnées x et y aléatoirement dans [-1; 1]
                           z est déduit par z = - (ax + by + d)
    on scale les valeurs de z pour aller dans [-1; 1]"""
    points = []
    for _ in range(n_plan):
        a, b, d = [random.uniform(-1, 1) for _ in range(3)]
        
        for _ in range(n_per_plan):
            x, y = [random.uniform(-1, 1) for _ in range(2)]
            z = -(a*x + b*x + d) + (bruit() if bruit is not None else 0)
            points.append([x,y,z])
        
    #todo
    raise NotImplementedError("remttre dans -1 1")
    return torch.tensor(points).float()
    
    #boules
    #gaussiennes
    #plan #droite #courbes


def _main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    ratio_train = .9
    n_models = 100
    n_train = int(n_models * ratio_train)
    
    n_points = 100
    space_dim = 3
    latent_size = 8
    
    nuages = [gen_sphere(2, n_points) for i in range(n_models)]
    
    #construction de notre modèle
    model = AutoEncoder3D(n_points, space_dim, latent_size)
    
    #apprentissage
    epochs = 50
    
    model.fit(nuages[:n_train], epochs, epochs//10, nuages[n_train:])
    
    for n in nuages:
        ax = plt.axes(projection='3d')
        y_pred = model(n).data
        ax.scatter(n[:, 0], n[:, 1], n[:, 2])
        ax.scatter(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2])
        plt.show()

def main():
    import ply
    path = "../AtlasNet/data/ShapeNetCorev2Normalized/02691156_normalised/d1a8e79eebf4a0b1579c3d4943e463ef.ply"
    a = ply.read_ply(path)
    print(a)
    path = "../AtlasNet/data/customShapeNet/02933112/ply/1a1b62a38b2584874c62bee40dcdc539.points.ply"
    a = ply.read_ply(path)
    print(a)


if __name__ == '__main__':
    _main()
