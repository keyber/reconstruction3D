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
    
    chosen_cat = 0
    n_mlp = 6
    grid_points = 10
    epochs = 30
    loss_factor_mode = 0
    
    if loss_factor_mode == 0:
        loss_factor = 1.0
    elif loss_factor_mode == 1:
        loss_factor = .001
    else:
        loss_factor = [.001]*(epochs*2//3) + [10.0]*(epochs//3)
    
    s = Segmentation(5)
    cloud = input_output.get_clouds([chosen_cat], nPerCat=1, ratio=.01)
    cloud = [Nuage(c, s) for c in cloud]
    latent_size = 25088
    latent = input_output.get_latent([chosen_cat], nPerCat=1, nPerObj=1)
    reconstructeur = Reconstructeur(n_mlp, latent_size, grid_points, s)
    list_pred = []
    loss_train, loss_test, times = fit_reconstructeur(reconstructeur, (latent, cloud), epochs,
                                                      ind_plotted=range(0, epochs, max(1,epochs//10)),
                                                      loss_factor=loss_factor,
                                                      list_predicted=list_pred)
    
    def update(i, pred, scattered):
        # met a jour le nuage de point prédit
        scattered[0].set_data(pred[i][:, 0], pred[i][:, 1])
        scattered[0].set_3d_properties(pred[i][:, 2])
        return scattered
    
    root = os.path.abspath(os.path.dirname(__file__))+"/../../outputs_animation/"
    file = "mlp" + str(n_mlp) + "_grid" + str(grid_points) + \
           "_lossmode"+ str(loss_factor_mode)+"_idcat"+str(chosen_cat)
    
    plt.plot(np.log10(loss_train))
    plt.title(file + "\n" + " ".join(map(str,times)))
    plt.savefig(root + "_loss_tmp")
    plt.close('all')
    
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_axis_off()
    ax.set_frame_on(False)
    ax.set_xlim3d([-1.0, 1.0]);ax.set_ylim3d([-1.0, 1.0]);ax.set_zlim3d([-1.0, 1.0])
    ax.set_title(file)
    l = cloud[0].liste_points.detach().numpy()
    # trace un nuage contenant seulement un point en (0,0,0) et le nuage ground truth
    scattered = [ax.plot([0], [0], [0], "g.")[0], ax.plot(l[:, 0], l[:, 1], l[:, 2], "r.")[0]]
    
    ani = animation.FuncAnimation(fig, update, epochs, fargs=(list_pred, scattered[:-1]), interval=500)
    plt.show()  # le dernier angle de vu est utilisé pour l'animation
    
    if os.path.exists(root + file + ".html"):
        input("overrite ?")
        os.remove(root+file+"_loss.png") # supprime le fichier des temps. (les autres fichiers sont écrasés automatiquement)
    os.rename(root+"_loss_tmp.png", root+file+"_loss.png")
    Writer = animation.writers['html']
    writer = Writer(fps=15, bitrate=1800)
    ani.save(root + file + ".html", writer=writer)
    with open(root + file + "_time", "w") as f:
        f.write(" ".join(map(str,times)))


if __name__ == '__main__':
    _main_anim()
