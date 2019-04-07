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
    
    chosen_cat = 0
    n_mlp = 6
    grid_points = 10
    epochs = 3
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
    res = fit_reconstructeur(reconstructeur, (latent, cloud), epochs,
                                                      loss_factor=loss_factor,
                                                      list_epoch_loss=range(0, epochs, max(1, epochs // 10)),
                                                      ind_cloud_saved=[0])
    

if __name__ == '__main__':
    _main_anim()
