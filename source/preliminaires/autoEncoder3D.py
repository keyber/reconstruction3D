import torch
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('./utils/')
import cloud_generation


def chamfer(Y, S):
    """return chamfer loss between
    the generated set G = {y[p] = f(x,p) for each p}
    and the real set S"""
    tmp = torch.sum(torch.pow(Y - S, 2), dim=(1,))
    #le MLP doit bien atteindre un Q
    loss1 = torch.min(tmp)
    
    #les Q doivent bien être atteint par le MLP
    loss2 = torch.sum(tmp)
    return torch.add(loss1, loss2)


def MSE(Y, S):
    """les points des deux ensembles doivent tous être en correspondance dans Y et S"""
    return torch.sum(torch.pow(Y - S, 2)) # somme des 3 coordonnées puis somme des distances


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
        # self.encoder[0] .weight.register_hook(lambda x:print("hook0:",torch.sum(torch.pow(x,2))))
        self.epochs = epochs
        self.verbose = False
    
    def forward(self, x):
        x = x.view(-1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, self.space_dim)
    
    def encode(self, x):
        return self.encoder(x.view(-1)).detach().numpy()
    
    def fit(self, x_train, x_test):
        """liste de nuages de points 3D"""
        optimizer = torch.optim.Adagrad(self.parameters(), lr=1e-2, lr_decay=.05)
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)
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
                # def print_graph(g, level=0):
                #     if g is None: return
                #     print('*' * level * 4, g)
                #     for subg in g.next_functions:
                #         print_graph(subg[0], level + 1)
                #
                # print_graph(loss.grad_fn)

                s_train += loss.item() / len(x_train)
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(self.encoder[0].weight[0][0], self.decoder[0].weight[0][0])
            
            list_loss_train.append(s_train)
            
            if t==0 or self.epochs<10 or (t+1) % (self.epochs // 10) == 0:
                self.eval()
                
                list_test_ind.append(t)
                
                s_test = sum(chamfer(self.forward(x), x.data).item() for x in x_test)/len(x_test) if len(x_test) else 0
                list_loss_test.append(s_test)
                
                if self.verbose:
                    print("time", t,
                          "loss train %.1e" % s_train,
                          "loss test %.1e" % s_test)
                self.train()
                
        if len(x_test):
            return list_loss_train, list_loss_test, list_test_ind
        return list_loss_train


def cross_validation(model_param, data, fold=5):
    if len(data) % fold != 0:
        raise ValueError("les " + str(len(data)) + " données ne sont pas partionables exactement en train test avec "
                         + str(fold) + "folds")
    
    batch_size = len(data) // fold
    mean_train = []
    std_train = []
    mean_test = []
    std_test = []
    for param in model_param:
        
        acc_train_test = []
        for i in range(fold):
            model = AutoEncoder3D(*param)
            
            data_test = data[batch_size * i: batch_size * (i + 1)]
            data_train = data[:batch_size * i] + data[batch_size * (i + 1):]
            
            p_train, p_test, test_ind = model.fit(data_train, data_test)
            
            acc_train_test.append([p_train[-1], p_test[-1]])
        
        acc_train_test = np.array(acc_train_test)
        mean_train.append(np.mean(acc_train_test[:, 0]))
        std_train.append(np.std(acc_train_test[:, 0]))
        mean_test.append(np.mean(acc_train_test[:, 1]))
        std_test.append(np.std(acc_train_test[:, 1]))
        print("score test final", mean_test[-1])
    
    return np.array(mean_train), np.array(std_train), np.array(mean_test), np.array(std_test)


def draw_cloud(ax, c):
    ax.scatter(c[:, 0], c[:, 1], c[:, 2])


def _main_fit_forward_draw(clouds, n_points, space_dim, n_draw):
    # FIT entrainement sur 100% pour l'affichage
    print("DRAW, fit en cours")
    latent_size = 25
    epochs = 100
    model = AutoEncoder3D(n_points, space_dim, latent_size, epochs)
    model.verbose = True
    print("loss moyenne finale", model.fit(clouds, []))
    
    for i, nuage in enumerate(clouds[:n_draw]):
        # FORWARD
        y_pred = model(nuage)
        mse = MSE(y_pred, nuage).item()
        print("nuage", i,
              "MSE", mse,
              "\t chamfer = MSE +", chamfer(y_pred, nuage).item() - mse)
        
        # DRAW
        ax = plt.axes(projection='3d')
        ax.set_title(str(i))
        draw_cloud(ax, nuage)
        draw_cloud(ax, y_pred.data)
        plt.show()


def _main_plot_cross_validation(clouds, n_points, space_dim, n_cross_validation):
    print("CROSS VALIDATION")
    epochs = [1, 10, 50]
    latent_sizes = [20]
    #cross validation pour les mesures
    params = [[(n_points, space_dim, latent_size, epoch) for epoch in epochs] for latent_size in latent_sizes]
    params = np.array(params).reshape((-1, 4))
    mean_train, std_train, mean_test, std_test = cross_validation(params, clouds, n_cross_validation)
    
    #proportion de data utilisé pour les test
    p = 1 / n_cross_validation
    n = len(clouds)
    
    #barres d'erreur à deux écarts types
    plt.errorbar(epochs, mean_train, yerr=2 * std_train / np.sqrt(n * (1 - p)), capsize=5)
    plt.errorbar(epochs, mean_test, yerr=2 * std_test / np.sqrt(n * p), capsize=5)
    plt.show()


def _main_clustering(clouds, n_classes, n_points, space_dim):
    print("CLUSTERING")
    latent_size = 2
    epochs = 100
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
    
    # fonction de répartition des gaussiennes
    std = .1
    loi = scipy.stats.truncnorm(-.5 / std, .5 / std, 0, std)
    
    n_points = 20
    
    range_gauss  = range(1,2)
    range_sphere = range(1,2)
    range_plan   = range(1,1)
    
    # ne marche pas : [lambda :gen_sphere(i, ...) for i in range(1,4)]
    cloud_generator = [(lambda i: lambda: cloud_generation.gen_sphere(i, n_points, radius_range=0, bruit=loi.rvs))(i) for i in range_gauss]
    cloud_generator += [(lambda i: lambda : cloud_generation.gen_sphere(i, n_points))(i) for i in range_sphere]
    cloud_generator += [(lambda i: lambda : cloud_generation.gen_plan(i, n_points))(i) for i in range_plan]
    
    n_clouds = 1 * n_cross_validation * len(cloud_generator)
    
    
    assert n_clouds % n_cross_validation == 0, "on doit pouvoir couper le dataset en n_cross_validation paquets"
    assert n_clouds // n_cross_validation % len(cloud_generator) == 0, "chaque paquet doit contenir autant de nuages de chaque classe"
    
    # générations des nuages
    clouds = []
    for _ in range(n_clouds // len(cloud_generator)):
        for m_gen in cloud_generator:
            clouds.append(m_gen())
    
    _main_plot_cross_validation(clouds, n_points, space_dim, n_cross_validation)
    
    _main_clustering(clouds, len(cloud_generator), n_points, space_dim)
    
    _main_fit_forward_draw(clouds, n_points, space_dim, n_clouds)


if __name__ == '__main__':
    _main()
