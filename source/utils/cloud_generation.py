import torch
import numpy as np
import random


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
    
    res = torch.tensor(points).float()
    res.requires_grad = False
    return res


def gen_plan(n_plan, n_per_plan, bruit=None):
    """plan : ax + by + cz + d = 0
    génère le plan en choisissant aléatoirement a, b et d (mais pas c)
    génère les coordonnées x et y aléatoirement dans [-1; 1]
                           z est déduit par z = - (ax + by + d)
    on scale les valeurs de z pour aller dans [-1; 1]"""
    points = []
    for _ in range(n_plan):
        a, b, d = [random.uniform(-1, 1) for _ in range(3)]
        
        for _ in range(n_per_plan):
            x, y = [random.uniform(-1, 1) for _ in range(2)]
            z = -(a * x + b * x + d) + (bruit[0](*bruit[1]) if bruit is not None else 0)
            points.append([x, y, z])
    points = np.array(points)
    
    # remet dans [-1, 1]
    points[:, 2] = points[:, 2] * (2 / (np.max(points[:, 2]) - np.min(points[:, 2]))) - 1
    res = torch.tensor(points).float()
    res.requires_grad = False
    return res
