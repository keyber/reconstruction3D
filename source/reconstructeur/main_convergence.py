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
    chosen_subset = [0]
    n_per_cat = 1
    clouds2 = input_output.get_clouds(chosen_subset, n_per_cat, ratio=.001)
    latent = input_output.get_latent(chosen_subset, n_per_cat,
                             nPerObj=1)  # /!\ attention: il faut que les fichiers sur le disque correspondent
    
    segmentation = Segmentation(5)
    clouds = [Nuage(x, segmentation) for x in clouds2]
    
    # plot_tailles(clouds)
    
    ratio_train = 1
    n_train = int(ratio_train * len(latent))
    
    indexes = list(range(len(latent)))
    #random.shuffle(indexes)
    train_x = [latent[i] for i in indexes[:n_train]]
    test_x = [latent[i] for i in indexes[n_train:]]
    train_y = [clouds[i] for i in indexes[:n_train]]
    test_y = [clouds[i] for i in indexes[n_train:]]
    train_y2 = [clouds2[i] for i in indexes[:n_train]]
    test_y2 = [clouds2[i] for i in indexes[n_train:]]
    
    n_mlp = 5
    latent_size = 25088  # défini par l'encodeur utilisé
    grid_points = 4
    epochs = 5
    grid_size = 1e0
    lr = 1e-4
    loss_factor = 1
    
    print("taille des nuages ground truth:", len(clouds2[0]))
    print("nombre de MLP:", n_mlp)
    print("résolution de la grille:", grid_points, "^2 =", grid_points ** 2)
    print("taille des nuages générés:", n_mlp, "*", grid_points ** 2, "=", n_mlp * grid_points ** 2)
    print("résolution segmentation:", segmentation.n_step, "^3 =", segmentation.n_step ** 3)
    
    reconstructeur1 = Reconstructeur(n_mlp, latent_size, grid_points, segmentation, quadratic=False)
    reconstructeur2 = Reconstructeur(n_mlp, latent_size, grid_points, segmentation, quadratic=True)
    
    for reconstructeur, param in zip([reconstructeur1, reconstructeur2], [train_y, train_y2]):
        # for reconstructeur, param in zip([reconstructeur1], [train_y]):
        # for reconstructeur, param in zip([reconstructeur2], [train_y2]):
        loss, t_loss, t_tot = fit_reconstructeur(reconstructeur, train_x, param, test_x, test_y, epochs, lr=lr,
                                                 grid_scale=grid_size, loss_factor=loss_factor)
        plt.plot(loss[1:])
        plt.show()
        print("temps: ")
        print("loss", sum(t_loss), t_loss)
        print("tot", sum(t_tot), t_tot)
        print("ratio", sum(t_loss) / sum(t_tot), t_loss / t_tot, "\n")
    
    print("répartition point parcourus histogramme :")
    print(np.histogram(Nuage.points_parcourus))
    print("moyenne", np.mean(Nuage.points_parcourus), "points parcourus")
    # print(segmentation.distances_2)
    
    output = reconstructeur1.forward(latent[0])
    output = [p.detach().numpy() for p in output.liste_points]
    input_output.write_clouds(os.path.abspath(os.path.dirname(__file__))+"/../../data/output_clouds", [output])


def plot_tailles(clouds):
    tailles = np.array([[len(x) for x in n.mat.reshape(-1)] for n in clouds], dtype=int).reshape(-1)
    plt.hist(tailles, bins=20, range=(1, max(tailles)))
    plt.show()


if __name__ == '__main__':
    _main()
