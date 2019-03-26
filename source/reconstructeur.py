import torch
from torch import nn
import numpy as np
import tSNE
import time
import matplotlib.pyplot as plt


class Segmentation:
    """défini la manière de partitionner l'espace 3D en cases"""
    
    def __init__(self, n_step):
        assert n_step % 2 == 1, "n_step doit être impair pour que le centre soit une cellule"
        self.dim = 3
        self.coo_min = -1
        self.coo_max = 1
        self.n_step = n_step
        self.n_step_div2 = n_step / 2
        
        # facteur pour passer de distances entre cases à des distances entre points
        self.factor_2 = (1 / self.n_step) ** 2
        
        # ensemble des déplacements(/voisins) possibles, ordonnés par leur distances
        self.neighbours, self.distances_2 = self._gen_neighbours()
        
        # sert à initialiser la matrice des nuages
        self.filler = np.frompyfunc(lambda x: torch.tensor([]), 1, 1)
    
    def _gen_neighbours(self):
        # génère tous les déplacements possibles de -n à n dans chaque dim
        r = range(self.n_step * 2)
        list_neighbours = np.array([[[(i, j, k) for i in r] for j in r] for k in r]).reshape(-1, 3)
        
        # shift de n pour centrer
        list_neighbours = list_neighbours - self.n_step
        
        # trie par distance au centre
        # il y a bcp d'équivalents
        # on regroupe les équivalents dans une même liste
        dict_neighbours = {}
        for voisin in list_neighbours:
            # la distance minimale entre des points des deux cases est égale à
            # la norme du vecteur entre les cases auquel on ENLEVE 1 DANS TOUTES LES DIMENSIONS NON NULLES
            # on laisse tout au carré
            d2 = np.sum(np.square(np.maximum(np.abs(voisin) - 1, 0)))
            
            # convertit la distance entre cases en une distance entre point
            d2 *= self.factor_2
            
            # une dimension égale à 1 devient équivalente à une dimension nulle
            # on ajoute un terme négligeable pour mettre en premiers les cases les plus proches du centre
            d2 += np.sum(np.abs(voisin) == 1) * 1e-6
            
            if d2 not in dict_neighbours:  # pas de pb d'arrrondi car on travaille sur des entiers, puis effectue les mêmes opérations
                dict_neighbours[d2] = []
            
            dict_neighbours[d2].append(voisin)
        
        dict_neighbours = {k: np.array(v) for (k, v) in dict_neighbours.items()}
        
        # calcule la liste triée des distances
        list_dist_2 = np.array(sorted(list(dict_neighbours.keys())))
        
        return dict_neighbours, list_dist_2
    
    def get_neighbours(self, cell, dist):
        """retourne la liste des cases à distance dist de cell"""
        # ajoute notre position pour calculer les coordonnées des voisins
        res = cell + self.neighbours[dist]
        
        # filtre les voisins qui sortent du tableau
        return res[np.all((0 <= res) & (res < self.n_step), axis=1)]
    
    def gen_matrix(self):
        """retourne un tableau 3D servant à stocker les points"""
        a = np.empty((self.n_step, self.n_step, self.n_step), dtype=list)
        return self.filler(a, a)
    
    def get_mat_coo(self, point):
        """retourne les coordonnées de la case du tableau contenant ce point"""
        res = np.asarray((point + 1) * self.n_step_div2, int)
        #<=> int((coo - self.coo_min) / (self.coo_max - self.coo_min) * self.n_step)
        
        # si une coordonnée est maximale (càd 1), le point sort du tableau, on le met dans la case d'avant
        return np.minimum(res, self.n_step - 1)
    
    def get_float_coo(self, point):
        """retourne les coordonnées de notre point dans la base des cases du tableau"""
        return np.minimum((point + 1) * self.n_step_div2, self.n_step - 1)


class Nuage:
    """stocke l'ensemble des points sous la forme
    d'une liste de points et
    d'un tableau à 3 dimensions"""
    
    def __init__(self, points, segmentation):
        self.segmentation = segmentation
        self.liste_points = points
        
        # crée le tableau
        self.mat = segmentation.gen_matrix()
        
        # remplit le tableau
        for p in self.liste_points:
            x, y, z = self.segmentation.get_mat_coo(p)
            self.mat[x, y, z] = torch.cat((self.mat[x, y, z], p.unsqueeze(0)))
    
    def recreate(self, points):
        """vide et re-rempli l'ancien tableau plutôt que de le supprimer et d'en allouer un nouveau"""
        self.liste_points = points
        
        # vide
        self.segmentation.filler(self.mat, self.mat)
        
        # remplit
        for p in self.liste_points:
            x, y, z = self.segmentation.get_mat_coo(p.detach())
            self.mat[x, y, z] = torch.cat((self.mat[x, y, z], p.unsqueeze(0)))
        
        return self
    
    # pour afficher des statistiques: liste du nombre de points parcourus pour chaque appel à get_closest
    points_parcourus = []
    
    def get_closest(self, point):
        # calcule la case dans laquelle tomberait le point
        case_coo = self.segmentation.get_mat_coo(point.detach().numpy())
        
        # le point le plus proche ne se trouve pas forcément dans la case la plus proche
        # un point plus proche peut encore être trouvé dans une case plus lointaine tant que
        # la distance à l'angle le plus proche de cette case est inférieur au min actuel
        point_coo = self.segmentation.get_float_coo(point.detach().numpy())
        
        cpt_point_parcourus = 0
        closest_point = torch.tensor(float("inf"))
        for d_2 in self.segmentation.distances_2:
            # stoppe la recherche si la distance minimale entre les angles des deux cases n'est pas meilleure
            if d_2 >= closest_point:
                break
            
            # considère toutes les cases non vides à distance d de notre case
            neighbours = [v for v in self.segmentation.get_neighbours(case_coo, d_2) if len(self.mat[v[0], v[1], v[2]])]
            if not len(neighbours): continue
            neighbours = np.array(neighbours)
            
            # calcule la distance entre notre point et le point des cases le plus proche
            cell_diff = point_coo - neighbours
            # on peut se déplacer dans la case en rajoutant un nombre compris entre 0 et 1
            cell_diff = cell_diff - np.minimum(1, np.maximum(0, cell_diff))
            cell_diff = np.sum(np.square(cell_diff), axis=1)
            # passe d'une distance entre cases à une distance entre point
            cell_diff = cell_diff * self.segmentation.factor_2
            
            # enlève les cases dont la distance min est trop grande
            ind_kept = np.where(cell_diff <= closest_point.item())
            if not len(ind_kept): continue
            cell_diff = cell_diff[ind_kept]
            neighbours = neighbours[ind_kept]
            
            # tri les cases par ordre croissant de distance
            ind_sorted = np.argsort(cell_diff)
            cell_diff = cell_diff[ind_sorted]
            neighbours = neighbours[ind_sorted]
            
            for real_d_2, v in zip(cell_diff, neighbours):
                if real_d_2 > closest_point:
                    break
                
                cpt_point_parcourus += len(self.mat[v[0], v[1], v[2]])
                
                # regarde si le point le plus proche parmi tous les points de la case est mieux
                candidat = torch.min(torch.sum(torch.pow(point - self.mat[v[0], v[1], v[2]], 2), dim=(1,)))
                if candidat < closest_point:
                    closest_point = candidat
            else:
                continue  # only executed if the inner loop did NOT break
            break  # only executed if the inner loop DID break
        
        Nuage.points_parcourus.append(cpt_point_parcourus)
        # return torch.sqrt(closest_point)
        return closest_point
    
    def chamfer(self, other, simplify_loss=False, squared_distances=False):
        # les points doivent bien être atteint par un MLP
        loss0 = torch.cat([self.get_closest(p).unsqueeze(0) for p in other.liste_points])
        if not squared_distances:
            loss0 = torch.sqrt(loss0)
        loss0 = torch.sum(loss0)
        
        # les MLP doivent bien atteindre un point
        if simplify_loss:
            loss1 = 0
        else:
            loss1 = torch.cat([other.get_closest(p).unsqueeze(0) for p in self.liste_points])
            if not squared_distances:
                loss1 = torch.sqrt(loss1)
            loss1 = torch.sum(loss1)
        
        return loss0, loss1


class Reconstructeur(nn.Module):
    def __init__(self, n_mlp, latent_size, n_grid_step, segmentation, quadratic):
        super().__init__()
        self._nuage_tmp = Nuage([], segmentation)
        self.input_dim = 2
        self.output_dim = 3
        self.n_grid_step = n_grid_step
        self.verbose = True
        self.n_mlp = n_mlp
        
        # s = [128, 64]
        s = [512, 256, 128]
        self.sizes = s
        
        # cf forward
        self.f0_a = nn.ModuleList([nn.Linear(latent_size, s[0]) for _ in range(n_mlp)])
        self.f0_b = nn.ModuleList([nn.Linear(self.input_dim, s[0]) for _ in range(n_mlp)])
        
        self.decodeurs = nn.ModuleList([nn.Sequential(
            nn.ReLU(),
            
            nn.Linear(s[0], s[1]),
            nn.ReLU(),
            
            nn.Linear(s[1], s[2]),
            nn.ReLU(),
            
            #on retourne un point dans l'espace de sortie
            nn.Linear(s[-1], self.output_dim),
            #tanh ramène dans [-1, 1], comme les nuages de notre auto-encodeur et de groueix
            nn.Tanh()
        ) for _ in range(n_mlp)])
        
        self.grid = self._gen_grid()
        
        self.loss = Reconstructeur.chamfer if quadratic else Nuage.chamfer
    
    def _gen_grid(self):
        """retourne une liste de n points 2D au bon format :
        array(tensor(coo_x, coo_y))
        les points forment un quadrillage du carré unitaire (il n'y a pas d'aléa)
        les tensors ont requires_grad=False car cela reviendrait à déformer la grille"""
        points_dim = [np.linspace(-1, 1, self.n_grid_step) for _ in range(self.input_dim)]
        points_dim = np.meshgrid(*points_dim)
        
        def f(*point_dim):
            v = torch.tensor(list(point_dim)).float()
            v.requires_grad = False
            return v
        
        return np.vectorize(f, otypes=[object])(*points_dim).reshape(-1)
    
    def forward(self, x0):
        # partie indépendante du sampling :
        y0_a = torch.cat([f(x0) for f in self.f0_a])
        y0_a = y0_a.reshape((self.n_mlp, 1, -1))
        
        # calcul pour tout f pour tout p
        y0_b = torch.cat([torch.cat([f(p) for p in self.grid]) for f in self.f0_b])
        y0_b = y0_b.reshape((self.n_mlp, len(self.grid), -1))
        
        # combine somme
        y0 = torch.add(y0_a, y0_b)
        
        # le reste
        res = torch.cat([torch.cat([f(x) for x in e]) for f, e in zip(self.decodeurs, y0)])
        res = res.reshape((self.n_mlp * len(self.grid), self.output_dim))
        if self.loss == Nuage.chamfer:
            res = self._nuage_tmp.recreate(res)
        return res
    
    @staticmethod
    def chamfer(Y, S, simplify_loss=False, squared_distances=False):
        """return chamfer loss between
        the generated set Y = {f(x, p) for each f, p}
        and the real pointcloud S corresponding to the latent vector x

        sum_F(sum_A) et min_A(min_F) correspondent à une simple somme ou min sur l'ensemble des données
        donc on représente l'ensemble des points générés Y comme une liste et non comme une matrice
        """
        normes = torch.cat([torch.cat([torch.sum(torch.pow(y - s, 2)).unsqueeze(0) for s in S]) for y in Y])
        normes = normes.reshape((len(Y), len(S)))
        
        loss0 = torch.min(normes, dim=0)[0]
        if not squared_distances:
            loss0 = torch.sqrt(loss0)
        loss0 = torch.sum(loss0)
        
        if simplify_loss:
            loss1 = 0
        else:
            loss1 = torch.min(normes, dim=1)[0]
            if not squared_distances:
                loss1 = torch.sqrt(loss1)
            loss1 = torch.sum(loss1)
        return loss0, loss1
        
        # # On sert uniquement des min selon les deux axes, on peut ne pas stocker la matrice pour éviter les problèmes de RAM
        # # listes des minimums sur chaque ligne et colonne
        # min_axe0 = [torch.tensor(float("+inf"))] * len(S)
        # min_axe1 = [torch.tensor(float("+inf"))] * len(Y)
        # # pour chaque case de la matrice, met à jour les deux minimums correspondants
        # for i, s in enumerate(S):
        #     for j, y in enumerate(Y):
        #         val = torch.norm(y - s)
        #         min_axe0[i] = torch.min(val, min_axe0[i])
        #         min_axe1[j] = torch.min(val, min_axe1[j])
        # return torch.sum(min_axe0), torch.sum(min_axe1)


def fit_reconstructeur(reconstructeur, x_train, y_train, x_test, y_test, epochs,
                       lr=1e-5, grid_scale=1.0, list_pred=None, simplify_loss=False, squared_distances=False):
    """grid_scale:
        les points 3D générés sont calculés à partir d'un échantillonage d'un carré 1*1,
        prendre un carré 100*100 revient à augmenter les coefficients du premier Linear sans pénaliser le modèle
    """
    reconstructeur.grid *= grid_scale
    optimizer = torch.optim.Adagrad(reconstructeur.parameters(), lr=lr, lr_decay=0.05)
    
    time_loss = np.zeros(epochs)
    time_tot = []
    list_loss = []
    
    for epoch in range(epochs):
        loss_train = 0
        t_tot = time.time()
        for x, y in zip(x_train, y_train):
            assert not x.requires_grad
            y_pred = reconstructeur.forward(x)
            
            if list_pred is not None:
                list_pred.append(y_pred.liste_points.detach().numpy().copy())
            
            t_loss = time.time()
            loss = reconstructeur.loss(y_pred, y, simplify_loss, squared_distances)
            time_loss[epoch] += time.time() - t_loss
            
            # print("loss", loss)
            loss = loss[0] + loss[1]
            loss_train += loss.item() / len(x_train)
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        time_tot.append(time.time() - t_tot)
        list_loss.append(loss_train)
        
        if reconstructeur.verbose and (epochs < 10 or epoch % (epochs // 10) == 0):
            reconstructeur.eval()
            
            s_test = sum(
                sum(reconstructeur.loss(reconstructeur.forward(x), y.data)).item() for (x, y) in zip(x_test, y_test))
            
            print("time", epoch,
                  "loss train %.4e" % loss_train,
                  "loss test %.2e" % s_test)
            reconstructeur.train()
        
        if loss_train / len(x_train) < 1e-8:
            #convergence finie, ca ne sert à rien de continuer
            break
    
    #apprentissage fini, passage en mode évaluation
    reconstructeur.eval()
    return list_loss, time_loss, time_tot


def draw_cloud(ax, c):
    ax.scatter(c[:, 0], c[:, 1], c[:, 2])


def plot_tailles(clouds):
    tailles = np.array([[len(x) for x in n.mat.reshape(-1)] for n in clouds], dtype=int).reshape(-1)
    plt.hist(tailles, bins=20, range=(1, max(tailles)))
    plt.show()


def _main_anim():
    import mpl_toolkits.mplot3d.axes3d as p3
    import matplotlib.animation as animation
    
    n_mlp = 8
    grid_points = 4
    epochs = 30
    simplify_loss = False
    squared_distances = False
    
    lr = 1e-4
    s = Segmentation(5)
    cloud = Nuage(tSNE.get_clouds([0], 1, ratio=.01)[0], s)
    latent_size = 25088
    latent = tSNE.get_latent([0], 1, nPerObj=1)[0]
    reconstructeur = Reconstructeur(n_mlp, latent_size, grid_points, s, quadratic=False)
    list_pred = []
    loss, t_loss, t_tot = fit_reconstructeur(reconstructeur, [latent], [cloud], [], [], epochs, lr=lr, list_pred=list_pred,
                                             simplify_loss=simplify_loss, squared_distances=squared_distances)
    
    def update(i, pred, scattered):
        # met a jour le nuage de point prédit
        scattered[0].set_data(pred[i][:, 0], pred[i][:, 1])
        scattered[0].set_3d_properties(pred[i][:, 2])
        return scattered
    
    root = "../outputs_animation/"
    file = "mlp" + str(n_mlp) + "_grid" + str(grid_points) + "_simplify" + str(simplify_loss) + "_squared" + str(squared_distances)
    
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


def _main():
    chosen_subset = [0]
    n_per_cat = 1
    clouds2 = tSNE.get_clouds(chosen_subset, n_per_cat, ratio=.001)
    latent = tSNE.get_latent(chosen_subset, n_per_cat,
                             nPerObj=1)  # /!\ attention: il faut que les fichiers sur le disque correspondent
    
    segmentation = Segmentation(5)
    clouds = [Nuage(x, segmentation) for x in clouds2]
    
    # plot_tailles(clouds)
    #tSNE.write_clouds("./data/output_clouds/ground_truth", [[p.detach().numpy() for p in clouds[0]]])
    
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
                                                 grid_scale=grid_size)
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
    tSNE.write_clouds("../data/output_clouds", [output])


def _test():
    s = Segmentation(5)
    c1 = Nuage(tSNE.get_clouds([0], 1, ratio=.01)[0], s)
    c2 = torch.tensor([[0., 0, 0], [1, 1, 1], [-1, -1, -1], [.2, .3, .0]])
    c2 = Nuage(c2, s)
    n_mlp = 2;latent_size = 25088;grid_points = 4;epochs = 5;lr = 1e-4
    latent = tSNE.get_latent([0], 1, nPerObj=1)[0]
    
    # équivalence des deux fonctions de loss (avec points écrits à la main)
    assert Nuage.chamfer(c1, c2) == Reconstructeur.chamfer(c1.liste_points, c2.liste_points)
    
    # presque équivalence de convergence
    # (utilise les mêmes nombres aléatoires pour avoir le même résultat)
    torch.manual_seed(0);np.random.seed(0)
    reconstructeur1 = Reconstructeur(n_mlp, latent_size, grid_points, s, quadratic=False)
    torch.manual_seed(0);np.random.seed(0)
    reconstructeur2 = Reconstructeur(n_mlp, latent_size, grid_points, s, quadratic=True)
    
    loss1, t_loss1, t_tot1 = fit_reconstructeur(reconstructeur1, [latent], [c1], [], [], epochs, lr=lr)
    loss2, t_loss2, t_tot2 = fit_reconstructeur(reconstructeur2, [latent], [c1.liste_points], [], [], epochs, lr=lr)
    
    # différences sûrement dues à des erreurs d'arrondi
    assert np.all(np.abs(np.array(loss1) - np.array(loss2)) < 1e-1)
    print("tests passés")


if __name__ == '__main__':
    # _test()
    _main_anim()
    exit()
    _main()

# torch.save(the_model.state_dict(), PATH)
# the_model = TheModelClass(*args, **kwargs)
# the_model.load_state_dict(torch.load(PATH))


"""todo list

calculer/mesurer coût en fonction de
  ground_truth_size
  n_mlp
  grid_point
  (epochs, n_clouds)
  loss chamfer ou simplifiée
  loss distances au carrées ou non


save 3D"""
