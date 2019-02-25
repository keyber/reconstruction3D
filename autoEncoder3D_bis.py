import torch
import numpy as np
import random
from torch.autograd import Variable


def chamfer(y, P, S):
    """return chamfer loss between
    the generated set G = {y[p] = f(x,p) for each p}
    and the real set S"""
    print(y[0])
    print(S[0])
    torch.norm(y[0] - S[0])
    torch.pow(torch.norm(y[0] - S[0]), 2)
    #les MLP doivent bien atteindre un Q
    loss = torch.sum(torch.tensor([torch.min(torch.tensor([torch.pow(torch.norm(y[p] - s), 2).item() for s in S])).item() for p in range(len(P))]))
    #les Q doivent bien être atteint par un MLP
    loss+= torch.sum(torch.tensor([torch.min(torch.tensor([torch.pow(torch.norm(y[p] - s), 2).item() for p in range(len(P))])).item() for s in S]))
    
    #for p in P:
    #    loss += min((y[p] - s) ** 2 for s in S)
    #for s in S:
    #    loss += min((y[p] - s) ** 2 for p in P)
    return loss


class AutoEncoder3D(torch.nn.Module):
    def __init__(self, D_in, space_dim, bottleneck_size, grid_points):
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
            torch.nn.Linear(bottleneck_size + space_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, D_in * space_dim),
            torch.nn.Tanh()
        )
        
        points_dim = [np.linspace(0, 1, grid_points) for _ in range(space_dim)]
        points_dim = np.meshgrid(*points_dim)
        
        def f(*point_dim):
            v = Variable(torch.tensor(list(point_dim)).float())
            v.require_grad = False
            return v
        
        #liste de n points 3D
        self.grid = np.vectorize(f, otypes=[object])(*points_dim).reshape(-1)
    
    def forward(self, x, p):
        raise NotImplementedError
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x, p):
        x = x.cat(p)
        return self.decoder(x)
    
    def fit(self, list_x, epochs, t_eval, test_x):
        """liste de nuages de points 3D"""
        
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)
        
        precisions = []
        
        for t in range(epochs):
            s_train = 0
            for x in list_x:
                latent = self.encode(x.view(-1).float())
                
                def f(point):
                    return self.decoder(torch.cat((latent, point)))
                
                y_pred = [f(p) for p in self.grid]
                #np.vectorize(f)(self.grid)
                
                loss = chamfer(y_pred, self.grid, x)
                s_train += loss.item()
                
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if t % t_eval == 0:
                self.eval()
                
                s_test = 0
                for x in test_x:
                    latent = self.encode(x.data)
                    decoder = np.vectorize(lambda point: self.decoder(latent, point).data)
                    y_pred = decoder(self.grid)
                    
                    loss = chamfer(y_pred, x.data, self.grid)
                    s_test += loss.item()
                
                print("time", t,
                      "loss train %.0e" % s_train,
                      "loss test %.0e" % s_test)
                self.train()
                
                if s_train / len(list_x) < 1e-10:
                    #convergence finie, ca ne sert à rien de continuer
                    break
        
        #apprentissage fini, passage en mode évaluation
        self.eval()
    
    def predict(self, list_x):
        res = []
        for x in list_x:
            y = self(x)
            res.append(max([0, 1], key=lambda i: y[i]))
        return res


def gen_sphere(n_spheres, n_per_spheres, n_dim=3, center_range=.5, radius_range=.5, bruit=None):
    assert radius_range + center_range <= 1
    
    points = []
    for _ in range(n_spheres):
        center = np.random.uniform(-center_range, center_range, n_dim)
        radius = random.uniform(0, radius_range)
        
        for _ in range(n_per_spheres):
            r = radius + (bruit() if bruit is not None else 0)
            orientation = np.random.uniform(-1, 1, n_dim)
            orientation *= r / np.sqrt(np.square(orientation).sum())
            points.append(center + orientation)
    
    return torch.tensor(points)
    
    #boules
    #gaussiennes
    #plan #droite #courbes


def _main():
    ratio_train = 1.0
    n = 1000
    n_train = int(n * ratio_train)
    
    space_dim = 3
    D_in = (n, space_dim)
    latent_size = 4
    grid_edge_size = 10
    
    nuages = [gen_sphere(1, n)]
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.axes(projection='3d')
    ax.scatter(nuages[0][:, 0], nuages[0][:, 1], nuages[0][:, 2])
    plt.show()
    """
    #construction de notre modèle
    model = AutoEncoder3D(n, space_dim, latent_size, grid_edge_size)
    
    #apprentissage
    epochs = 1000  #aura convergé avant
    
    model.fit(nuages[:n_train], epochs, 10, nuages[n_train:])


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
