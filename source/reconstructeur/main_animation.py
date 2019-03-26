import numpy as np
import matplotlib.pyplot as plt
from segmentation import Segmentation
from nuage import Nuage
from model import Reconstructeur, fit_reconstructeur
import os
import sys
sys.path.append('./utils/')
import input_output

def _main_anim():
    import mpl_toolkits.mplot3d.axes3d as p3
    import matplotlib.animation as animation
    
    n_mlp = 5
    grid_points = 3
    epochs = 10
    loss_factor = 1
    squared_distances = True
    
    lr = 1e-4
    s = Segmentation(5)
    cloud = Nuage(input_output.get_clouds([0], 1, ratio=.01)[0], s)
    latent_size = 25088
    latent = input_output.get_latent([0], 1, nPerObj=1)[0]
    reconstructeur = Reconstructeur(n_mlp, latent_size, grid_points, s, quadratic=False)
    list_pred = []
    loss, t_loss, t_tot = fit_reconstructeur(reconstructeur, [latent], [cloud], [], [], epochs, lr=lr,
                                             list_pred=list_pred,
                                             loss_factor=loss_factor, squared_distances=squared_distances)
    
    def update(i, pred, scattered):
        # met a jour le nuage de point prédit
        scattered[0].set_data(pred[i][:, 0], pred[i][:, 1])
        scattered[0].set_3d_properties(pred[i][:, 2])
        return scattered
    
    root = os.path.abspath(os.path.dirname(__file__))+"/../../outputs_animation/"
    file = "mlp" + str(n_mlp) + "_grid" + str(grid_points) + "_loss_factor" + str(loss_factor) + "_squared" + str(
        squared_distances)
    
    plt.plot(np.log10(loss))
    plt.title(file)
    plt.savefig(root + file + "_loss")
    plt.close('all')
    
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_axis_off()
    ax.set_frame_on(False)
    ax.set_xlim3d([-1.0, 1.0]);ax.set_ylim3d([-1.0, 1.0]);ax.set_zlim3d([-1.0, 1.0])
    ax.set_title(file)
    l = cloud.liste_points.detach().numpy()
    scattered = [ax.plot([0], [0], [0], "g.")[0], ax.plot(l[:, 0], l[:, 1], l[:, 2], "r.")[0]]
    
    ani = animation.FuncAnimation(fig, update, epochs, fargs=(list_pred, scattered[:-1]), interval=500)
    plt.show()  # le dernier angle de vu est utilisé pour l'animation
    Writer = animation.writers['html']
    writer = Writer(fps=15, bitrate=1800)
    ani.save(root + file + ".html", writer=writer)
    with open(root + file + "_time", "w") as f:
        f.write("total " + str(sum(t_tot) / len(t_tot)) + "\t")
        for t in t_tot:
            f.write(str(t) + ",")
        f.write("\n")
        f.write("loss " + str(sum(t_loss) / len(t_loss)) + "\t")
        for t in t_loss:
            f.write(str(t) + ",")


if __name__ == '__main__':
    _main_anim()
