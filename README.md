# Géométrie dans l'espace - Reconstruction 3D

## Description
Le but de ce PLDAC est de travailler sur la reconstruction de structure 3D à partir d'image, avec comme supervision des nuages de points. Il s'agit de mettre en place une architecture capable d'analyser des images et d'estimer la forme des objets en sortie. Il faudra donc prendre en main des architectures de réseaux de neurones pour traiter les images puis bâtir une architecture au dessus.
L'idée est de partir des [travaux de Thibault Groueix](http://imagine.enpc.fr/~groueixt/).

1) Prise en main des architectures images et des données
2) Construction d'un modèle prédictif
3) Affinage du modèle et mise en valeur des résultats (un peu de codage OpenGL)


## TODO
1) Comprendre l'article, bibliographie
2) Prise en main de la chaîne de traitement
    - Téléchargement des mêmes données que l'auteur utilise depuis le cloud de l'ENCP.
    - Installation des librairies nécessaires
    - Chargement d'un modèle (vgg/ResNet) sur PyTorch (lecture de tutoriels PyTorch)
    - Exécuter l'algorithme forward du modèle (image -> vecteur latent) sur un sous-ensemble d'image et vérifier son bon fonctionnement en affichant le [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) des vecteurs latents.
    - Afficher des nuages de points (matplotlib et/ou logiciel annexe)
    - Afficher des nuages de points avec PyOpenGL / [Open3D](http://www.open3d.org/): L'affichage final se fera uniquement en Open3D. 
3) Entraîner un réseau de neurones simple classifiant les objets à partir de leur représentation latente. Le réseau de neurones est composé d'une seule couche, cette couche est linéaire.
4)
   1) Définir un auto encoder nuage de points 3D :
      - entrée "3N" : un ensemble de n points dans \[-1; 1]<sup>3</sup>
      - encodeur : 3N -> FC 512 -> ReLU -> FC 128 -> ReLU -> FC 2
      - (on récupère un vecteur latent de taille 2)
      - décodeur  : 2 -> FC 128 -> ReLU -> FC 512 -> ReLU -> FC 3N -> tanh
      - sortie de même dimension que l'entrée
   2) Définir la fonction de coût "distance de Chamfer" en ne se servant que d'un seul MLP
   3) Générer des nuages de points pour entraîner le réseau
5) Etoffer travail précédent:
   - calculer MSE en plus de Chamfer
   - visualiser résultat en fonction de coût
   - coût en fonction de taille du vecteur latent
6) Implémentation de la technique de l'échantillonage du carré unitaire
7) Implémentation du décodeur avec le vecteur latent de VGG 
99) Développement selon nos envies
100) On pourra voir les autres méthodes sans les implémenter.


## Carnet de bord
- 09/01 première rencontre

- 23/01
  - On a choisi le **framework** de deep learning **PyTorch** plutôt que TensorFlow ou Keras.
  - Les CNN (Convolutional Neural Network) sont des réseaux de neurones spécialement conçus pour travailler sur des images.
 [ImageNet](http://www.image-net.org/) est une base de données d'images contenant plus de 10<sup>4</sup> catégories et 10<sup>2</sup> images par catégories.
 Parmi les différents **CNN** existants, on a choisi **VGG** (développé par Visual Geometry Group) pour ses performances et sa simplicité.
  - Via PyTorch, on a accès à différentes versions de réseaux VGG préentrainés sur ImageNet. On se servira par exemple de *VGG16* (réseau de neurones à 16 couches, sans "batch normalisation") https://pytorch.org/docs/master/torchvision/models.html#torchvision.models.vgg16.

- 29/01
  - Alternativement à VGG, on pourra se servir de ResNet (CNN avec une architecture complexe mais utilisé par Groueix)
  - Précision de tâches du cahier des charges (t-SNE)

- 08/02
  - Visualisation t-SNE validée
  - Précision de tâches du cahier des charges (entraînement d'un réseau de neurone simple, openGL)


- 13/02
  - Les résultats obtenus par notre réseau de neurone entrainé (dont nous sommes incertains) sont cohérents.
  - Précision de tâches du cahier des charges (implémentation des fonctions de coût, génération 3D, openGL)

- 19/02
  - Explication de l'article de Groueix
  - Précision de tâches du cahier des charges (entraînement d'un autoencoder de nuage de points avec la distance de Chamfer)

- 26/02
  - Précision de tâches du cahier des charges (liens qualitatif / quantitatif, chamfer / taille latente, MSE / taille_latente)

- 05/03
  - Analyse du compte rendu actuel : t-SNE OK. auto-encodeur 3D à mieux expliquer, tracer erreur en fonction du nombre d'epoch. Afficher des statistiques par classes de nuages. Visualiser des regroupements en clusters en utilisant un espace latent de taille 2.
  - Analyse du code du reconstructeur.
  - Face au problème de performances de la distance de Chamfer, Google Colab a été mentionné.

- 12/03
  - Visualisation du auto-encoder à expliquer
  - Visualiser plan vs sphere
  - Chamfer O(n) ne suffit pas, il faut utiliseer Google Colab
  - Chamfer doit utiliser des fonctions de pytorch et non de numpy
  - L'affichage des résultats se fera en Open3D pour pallier les probèmes rencontrés avec PyOPenGL.
  - Importer / coder une des techniques de reconstruction 3D existantes afin de l'utiliser comme une baseline pour l'évaluation de notre modèle.

- 26/03
  - manière optimisée de calcul du forward et chamfer acceptés.
  - faire varier l'importance du deuxième facteur de chamfer, faire la moyenne plutôt que la somme des min

- 02/04

Projet effectué à l'UPMC dans le master DAC pour le M1S2 de l'année 2018/2019.
Encadrants : Vincent Guigue & Nicolas Baskiotis.
