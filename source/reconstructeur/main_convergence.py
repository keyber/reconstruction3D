import numpy as np
from kdtree import Nuage
from model import Reconstructeur, fit_reconstructeur
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import os
import sys
sys.path.append('./utils/')
import input_output
from sklearn.model_selection import train_test_split


def _main():
    chosen_subset = [1]
    n_per_cat = 4
    sample_size = 300
    eps_ground_truth = 1e-5 # algorithme eps approché: la distance obtenue est inférieure à (1 + eps) * opt
    clouds = input_output.get_clouds(chosen_subset, n_per_cat, ratio=10*sample_size)
    latent = input_output.get_latent(chosen_subset, n_per_cat, nPerObj=1)  # /!\ attention: il faut que les fichiers sur le disque correspondent
    clouds = [Nuage(x, eps=eps_ground_truth) for x in clouds]
    
    ratio_test = .2
    train_x, test_x, train_y, test_y = train_test_split(latent, clouds, test_size=ratio_test)
    print("train", len(train_y), "test", len(test_y))
    
    latent_size = 25088  # défini par l'encodeur utilisé
    grid_size = 1e0
    n_mlp = 8
    grid_points = 6
    epochs = 10
    lr = 1e-4
    loss_factor_mode = 0

    if loss_factor_mode == 0:
        loss_factor = 1.0
    elif loss_factor_mode == 1:
        loss_factor = .001
    else:
        loss_factor = [.001] * (epochs * 2 // 3) + [10.0] * (epochs // 3)

    print("taille des nuages ground truth:", sample_size, "extraits de", len(clouds[0].points))
    print("nombre de MLP:", n_mlp)
    print("résolution de la grille:", grid_points, "^2 =", grid_points ** 2)
    print("taille des nuages générés:", n_mlp, "*", grid_points ** 2, "=", n_mlp * grid_points ** 2)
    
    reconstructeur = Reconstructeur(n_mlp, latent_size, grid_points, Nuage.chamfer)
    
    list_epoch_loss = range(0, epochs, max(1, epochs//10))
    ind_cloud_saved = {0, len(clouds)-2, len(clouds)-1}
    res = fit_reconstructeur(reconstructeur, (train_x, train_y), epochs, sample_size=sample_size,
                             lr=lr, grid_scale=grid_size, loss_factor=loss_factor,
                             test=(test_x, test_y),
                             list_epoch_loss=list_epoch_loss,
                             ind_cloud_saved=ind_cloud_saved)
    
    root = os.path.abspath(os.path.dirname(__file__)) + "/../../outputs_animation/"
    file = "mlp" + str(n_mlp) + "_grid" + str(grid_points) + "_lossmode" + str(loss_factor_mode) + "_idcat"\
           + (str(chosen_subset[0] if len(chosen_subset)==1 else chosen_subset))
    
    save(root, file, res, epochs, list_epoch_loss, clouds, ind_cloud_saved)


def update(i, pred, scattered):
    # met a jour le nuage de point prédit
    scattered.set_data(pred[i][:, 0], pred[i][:, 1])
    scattered.set_3d_properties(pred[i][:, 2])
    return scattered


def save(root, file, res, epochs, list_epoch_loss, clouds, ind_cloud_saved):
    plt.plot(range(epochs), np.log10(res["loss_train"])[:,0], color='#BBBBFF')
    plt.plot(range(epochs), np.log10(res["loss_train"])[:,1], color='#BBFFBB')
    plt.plot(range(epochs), np.log10(np.sum(res["loss_train"],axis=1)), color='#FFBBBB')
    if res["loss_test"]:
        plt.plot(list_epoch_loss, np.log10(np.sum(res["loss_test"], axis=1)), color='#BB0000', label="test")
        plt.plot(list_epoch_loss, np.log10(res["loss_test"])[:,0], color='#0000BB', label="atteint")
        plt.plot(list_epoch_loss, np.log10(res["loss_test"])[:,1], color='#00BB00', label="dépasse")
    plt.legend()
    plt.title(file + "\n" +
              "tot {0:.2f}, loss {1:.2f}, ratio {2:.2f}".format(*res["time"]))
    plt.savefig(root + "_loss_tmp")
    plt.show()
    
    for ind_c in ind_cloud_saved:
        plt.close("all")
        fig = plt.figure('figure')
        ax = p3.Axes3D(fig)
        ax.set_axis_off()
        ax.set_frame_on(False)
        ax.set_xlim3d([-1.0, 1.0]);ax.set_ylim3d([-1.0, 1.0]);ax.set_zlim3d([-1.0, 1.0])
        ax.set_title(file)
        
        # trace un nuage contenant seulement un point en (0,0,0)
        scattered = ax.plot([0], [0], [0], "g.")[0]
        
        # trace le nuage ground truth
        l = clouds[ind_c].points.detach().numpy()
        ax.plot(l[:, 0], l[:, 1], l[:, 2], "r.")
        
        ani = animation.FuncAnimation(fig, update, epochs, fargs=(res["predicted"][ind_c], scattered), interval=500)
        plt.show()  # le dernier angle de vu est utilisé pour l'animation
    
        Writer = animation.writers['html']
        writer = Writer(fps=15, bitrate=1800)
        ani.save(root + file + ".html", writer=writer)


if __name__ == '__main__':
    _main()

"""
visualiser inférences àp même sampling
resortir un nuage avec bcp plus de points.
essayer de les relier.

minibatch converge bien ou moyenne ?
X formes simples (ex spheres)
"""
