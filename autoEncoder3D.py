import torch
import numpy as np
import random
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def chamfer(Y, S):
    """return chamfer loss between
    the generated set G = {y[p] = f(x,p) for each p}
    and the real set S"""
    tmp = torch.pow(torch.norm(Y - S, keepdim=True), 2)
    
    #le(s) MLP doivent bien atteindre un Q
    loss1 = torch.min(tmp)  # NEGLIGEABLE DE 4 ORDRES DE GRANDEURS
    #les Q doivent bien être atteint par un MLP
    loss2 = torch.sum(tmp)
    return loss1 + loss2


def MSE(Y, S):
    """les points des deux ensembles doivent tous être en correspondance dans Y et S"""
    loss = torch.sum(torch.pow(torch.norm(Y - S, dim=1), 2))
    return loss


class AutoEncoder3D(torch.nn.Module):
    def __init__(self, D_in, space_dim, bottleneck_size, epochs):
        super().__init__()
        self.space_dim = space_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(D_in * space_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, bottleneck_size),
            torch.nn.ReLU()
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, D_in * space_dim),
            torch.nn.Tanh()
        )
        self.epochs = epochs
        self.verbose = False
    
    def forward(self, x):
        x = x.view(-1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, self.space_dim)
    
    def encode(self, x):
        return self.encoder(x.view(-1)).detach().numpy()
    
    def fit(self, x_train, x_test, lr=1e-4):
        """liste de nuages de points 3D"""
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        list_loss_train = []
        list_loss_test = []
        list_test_ind = []
        
        for t in range(self.epochs):
            s_train = 0
            for x in x_train:
                assert not x.requires_grad
                
                y_pred = self.forward(x)
                assert y_pred.requires_grad
                
                loss = chamfer(y_pred, x)
                
                s_train += loss.item() / len(x_train)
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            list_loss_train.append(s_train)
            
            if self.verbose and (t==0 or (t+1) % (self.epochs // 10) == 0):
                self.eval()
                
                list_test_ind.append(t)
                
                s_test = sum(chamfer(self.forward(x), x.data).item() for x in x_test)/len(x_test) if len(x_test) else 0
                list_loss_test.append(s_test)
                
                print("time", t,
                      "loss train %.1e" % s_train,
                      "loss test %.1e" % s_test)
                self.train()
                
                if s_train / len(x_train) < 1e-8:
                    #convergence finie, ca ne sert à rien de continuer
                    break
        
        #apprentissage fini, passage en mode évaluation
        self.eval()
        if len(x_test):
            return list_loss_train, list_loss_test, list_test_ind
        return list_loss_train


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


def cross_validation(model_param, data, fold=5, lr=1e-4):
    if len(data) % fold != 0:
        raise ValueError("les " + str(len(data)) + " données ne sont pas partionables exactement en train test avec "
                         + str(fold) + "folds")
    
    batch_size = len(data) // fold
    mean_train = []
    std_train = []
    mean_test = []
    std_test = []
    for param in model_param:
        
        model = AutoEncoder3D(*param)
        
        acc_train_test = []
        for i in range(fold):
            data_test = data[batch_size * i: batch_size * (i + 1)]
            data_train = data[:batch_size * i] + data[batch_size * (i + 1):]
            
            p_train, p_test = model.fit(data_train, data_test, lr)
            
            acc_train_test.append([p_train[-1], p_test[-1]])
        
        acc_train_test = np.array(acc_train_test)
        mean_train.append(np.mean(acc_train_test[:, 0]))
        std_train.append(np.std(acc_train_test[:, 0]))
        mean_test.append(np.mean(acc_train_test[:, 1]))
        std_test.append(np.std(acc_train_test[:, 1]))
        print(mean_test[-1])
    
    return np.array(mean_train), np.array(std_train), np.array(mean_test), np.array(std_test)


def draw_cloud(ax, c):
    ax.scatter(c[:, 0], c[:, 1], c[:, 2])


def _main_fit_forward_draw(clouds, param, n_draw):
    # FIT entrainement sur 100% pour l'affichage
    print("fit en cours")
    model = AutoEncoder3D(*param)
    print("loss moyenne finale", model.fit(clouds, []))
    
    for i, nuage in enumerate(clouds[:n_draw]):
        # FORWARD
        y_pred = model(nuage)
        mse = MSE(y_pred, nuage).item()
        print("modèle", i,
              "MSE", mse,
              "\t chamfer = MSE +", chamfer(y_pred, nuage).item() - mse)
        
        # DRAW
        ax = plt.axes(projection='3d')
        ax.set_title(str(i))
        draw_cloud(ax, nuage)
        draw_cloud(ax, y_pred.data)
        plt.show()


def _main_plot_cross_validation(clouds, n_points, space_dim, n_cross_validation):
    epochs = [1, 5, 10, 20]
    latent_sizes = [2]
    #cross validation pour les mesures
    params = [[(n_points, space_dim, latent_size, epoch) for epoch in epochs] for latent_size in latent_sizes]
    params = np.array(params).reshape((-1, 4))
    mean_train, std_train, mean_test, std_test = cross_validation(params, clouds, n_cross_validation, lr=1e-5)
    
    #proportion de data utilisé pour les test
    p = 1 / n_cross_validation
    n = len(clouds)
    
    #barres d'erreur à deux écarts types
    plt.errorbar(epochs, mean_train, yerr=2 * std_train / np.sqrt(n * (1 - p)), capsize=5)
    plt.errorbar(epochs, mean_test, yerr=2 * std_test / np.sqrt(n * p), capsize=5)
    plt.show()


def _main_clustering(clouds, n_classes, n_points, space_dim):
    latent_size = 2
    epochs = 500
    model = AutoEncoder3D(n_points, space_dim, latent_size, epochs)
    model.verbose = True
    n = int(.8*len(clouds))
    loss_train, loss_test, test_ind = model.fit(clouds[:n], clouds[n:])
    plt.plot(loss_train)
    plt.plot(test_ind, loss_test)
    plt.show()
    
    a = np.array([model.encode(c) for c in clouds[:n]])
    plt.scatter(a[:,0], a[:, 1], c=['r','g','b','c','m'][:n_classes]*(n//n_classes))
    plt.show()

def _main():
    space_dim = 3
    n_cross_validation = 3
    #fonction de répartition des gaussiennes
    std = .1
    loi = scipy.stats.truncnorm(-.5 / std, .5 / std, 0, std)
    
    n_points = 20
    
    range_gauss  = range(1,3)
    range_sphere = range(1,1)
    
    #ne marche pas : [lambda :gen_sphere(i, ...) for i in range(1,4)]
    cloud_generator = [(lambda i: lambda: gen_sphere(i, n_points, radius_range=0, bruit=loi.rvs))(i) for i in range_gauss]
    cloud_generator += [(lambda i: lambda : gen_sphere(i, n_points))(i) for i in range_sphere]
    
    n_clouds = 20 * n_cross_validation * len(cloud_generator)
    
    
    assert n_clouds % n_cross_validation == 0, "on doit pouvoir couper le dataset en n_cross_validation paquets"
    assert n_clouds // n_cross_validation % len(cloud_generator) == 0, "chaque paquet doit contenir autant de nuages de chaque classe"
    
    # générations des nuages
    clouds = []
    for _ in range(n_clouds // len(cloud_generator)):
        for m_gen in cloud_generator:
            clouds.append(m_gen())
    
    #ax = plt.axes(projection='3d')
    #draw_cloud(ax, clouds[0])
    #plt.show()
    
    _main_clustering(clouds, len(cloud_generator), n_points, space_dim)
    
    # _main_fit_forward_draw(clouds, (n_points, space_dim, 2, 50), 400*len(cloud_generator))

    # _main_plot_cross_validation(clouds, n_points, space_dim, n_cross_validation)

if __name__ == '__main__':
    _main()

"""
rajouter une couche 64

stats par type de forme err std

visualiser regroupement en clusters avec espace latent de taille 2

google colab"""
