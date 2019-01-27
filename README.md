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
    - Téléchargement des données depuis le cloud de l'ENCP.
3) Développement selon nos envies (lecture de tutoriels PyTorch peut-être nécessaire)
4) On pourra voir les autres méthodes sans les implémenter.


## Carnet de bord
- 09/01 première rencontre

- 23/01
  - On a choisi le **framework** de deep learning **PyTorch** plutôt que TensorFlow ou Keras.
  - Les CNN (Convolutional Neural Network) sont des réseaux de neurones spécialement conçus pour travailler sur des images.
 [ImageNet](http://www.image-net.org/) est une base de données d'images contenant plus de 10^4 catégories et 10^2 images par catégories.
 Parmi les différents **CNN** existants, on a choisi **VGG** (développé par Visual Geometry Group) pour ses performances et sa simplicité.
  - Via PyTorch, on a accès à différentes versions de réseaux VGG préentrainés sur ImageNet. On se servira par exemple de *VGG16* (réseau de neurones à 16 couches, sans "batch normalisation") https://pytorch.org/docs/master/torchvision/models.html#torchvision.models.vgg16.

- 29/01


 
 
 
Projet effectué à l'UPMC dans le master DAC pour le M1S2 de l'année 2018/2019.

Encadrants : Vincent Guigue & Nicolas Baskiotis.
