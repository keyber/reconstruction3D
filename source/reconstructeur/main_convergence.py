import numpy as np
from segmentation import Segmentation
from nuage import Nuage
from model import Reconstructeur, fit_reconstructeur
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('./utils/')
import input_output


def _main():
    chosen_subset = [2]
    n_per_cat = 10
    clouds = input_output.get_clouds(chosen_subset, n_per_cat, ratio=.01)
    latent = input_output.get_latent(chosen_subset, n_per_cat,
                             nPerObj=1)  # /!\ attention: il faut que les fichiers sur le disque correspondent
    segmentation = Segmentation(5)
    clouds = [Nuage(x, segmentation) for x in clouds]
    
    # plot_tailles(clouds)
    
    ratio_train = .8
    n_train = int(ratio_train * len(latent))
    
    indexes = list(range(len(latent)))
    #random.shuffle(indexes)
    train_x = [latent[i] for i in indexes[:n_train]]
    test_x = [latent[i] for i in indexes[n_train:]]
    train_y = [clouds[i] for i in indexes[:n_train]]
    test_y = [clouds[i] for i in indexes[n_train:]]
    
    n_mlp = 10
    latent_size = 25088  # défini par l'encodeur utilisé
    grid_points = 4
    epochs = 30
    grid_size = 1e0
    lr = 1e-4
    loss_factor = [.001]*(epochs*2//3) + [10.0]*(epochs//3)
    
    print("taille des nuages ground truth:", len(clouds[0].liste_points))
    print("nombre de MLP:", n_mlp)
    print("résolution de la grille:", grid_points, "^2 =", grid_points ** 2)
    print("taille des nuages générés:", n_mlp, "*", grid_points ** 2, "=", n_mlp * grid_points ** 2)
    print("résolution segmentation:", segmentation.n_step, "^3 =", segmentation.n_step ** 3)
    
    reconstructeur = Reconstructeur(n_mlp, latent_size, grid_points, segmentation, quadratic=False)
    
    ind_plotted = range(0, epochs, max(1, epochs//10))
    loss_train, loss_test, times = fit_reconstructeur(reconstructeur, (train_x, train_y), epochs,
                                                      lr=lr, grid_scale=grid_size, loss_factor=loss_factor,
                                                      test=(test_x, test_y),
                                                      ind_plotted=set(ind_plotted))
    plt.plot(range(epochs), np.log10(loss_train))
    plt.plot(ind_plotted, np.log10(loss_test))
    plt.show()
    
    print("répartition point parcourus histogramme :")
    print(np.histogram(Nuage.points_parcourus))
    print("moyenne", np.mean(Nuage.points_parcourus), "points parcourus")
    # print(segmentation.distances_2)
    
    output = [reconstructeur.forward(x).detach().numpy() for x in latent]
    input_output.write_clouds(os.path.abspath(os.path.dirname(__file__))+"/../../data/output_clouds", output)


def plot_tailles(clouds):
    tailles = np.array([[len(x) for x in n.mat.reshape(-1)] for n in clouds], dtype=int).reshape(-1)
    plt.hist(tailles, bins=20, range=(1, max(tailles)))
    plt.show()


if __name__ == '__main__':
    _main()
