import torch
from model import Reconstructeur
from kdtree import Nuage
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import sys
sys.path.append('./utils/')
import input_output


def _main_inference():
    chosen_subset = [0]
    n_per_cat = 10
    sample_size = 1000
    clouds = input_output.get_clouds(chosen_subset, n_per_cat, size=sample_size)
    latent = input_output.get_latent(chosen_subset, n_per_cat,
                                     nPerObj=1)  # /!\ attention: il faut que les fichiers sur le disque correspondent
    latent_size = 25088
    loaded_name = "trained_network_mlp6_grid4_lossmode2_idcat0_clouds2_epoch15"
    n_mlp = int(loaded_name.split("mlp")[1].split("_")[0])
    
    # en inférence, on n'utilise pas forcément la même résolution de grille
    # grid_points = int(loaded_name.split("grid")[1].split("_")[0])
    grid_points = 16
    reconstructeur = Reconstructeur(n_mlp, latent_size, grid_points, Nuage.chamfer)
    reconstructeur.load_state_dict(torch.load(loaded_name))
    
    for x, ground_truth in zip(latent, clouds):
        pred = reconstructeur.forward(x)
        fig = plt.figure('figure')
        ax = p3.Axes3D(fig)
        ax.set_axis_off()
        ax.set_frame_on(False)
        ax.set_xlim3d([-1.0, 1.0])
        ax.set_ylim3d([-1.0, 1.0])
        ax.set_zlim3d([-1.0, 1.0])

        # trace le nuage ground truth
        l = ground_truth.detach().numpy()
        ax.plot(l[:, 0], l[:, 1], l[:, 2], "g,")
        l = pred.detach().numpy()
        ax.plot(l[:, 0], l[:, 1], l[:, 2], "r.")

        plt.show()  # le dernier angle de vu est utilisé pour l'animation


if __name__ == '__main__':
    _main_inference()

"""
visualiser inférences àp même sampling
resortir un nuage avec bcp plus de points.
essayer de les relier.

minibatch converge bien ou moyenne ?
X formes simples (ex spheres)
"""
