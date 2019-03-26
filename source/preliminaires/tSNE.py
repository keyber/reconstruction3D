from sklearn import manifold
from sklearn import decomposition
import matplotlib.pyplot as plt
import sys
sys.path.append('./utils/')
import input_output


def tSNE(vectors, nCat):
    #from mpl_toolkits.mplot3d import Axes3D
    #ax = plt.axes()#projection='3d')
    print("tSNE  input dimension", len(vectors[0]))
    
    n_components = min(len(vectors), 50)
    
    pca = decomposition.PCA(n_components=n_components)
    
    principalComponent = pca.fit_transform([v.numpy() for v in vectors])
    
    print("après PCA", type(principalComponent), principalComponent.shape)
    
    colors = [(r/255, g/255, b/255) for r, g, b in
              [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
               (0, 0, 0),  (128, 128, 128), (255, 165, 0), (210, 180, 140),
               (192, 192, 192), (128, 0, 0), (0, 128, 0), (0, 0, 128), (165, 42, 42)]]
    perCat = len(vectors) // nCat
    
    for perplexity in [5, 15, 50]:
        for learnig_rate in [30, 100, 200]:
            plt.figure()
            tsne = manifold.TSNE(n_components=2, learning_rate=learnig_rate,
                                 perplexity=perplexity, n_iter=100000)
            Y = tsne.fit_transform(principalComponent)
            
            for cat in range(nCat):
                plt.scatter(Y[cat*perCat:(cat+1)*perCat, 0], Y[cat*perCat:(cat+1)*perCat, 1],
                            c=[colors[cat % len(colors)]])


def _main():
    chosenSubSet = [0, 1]
    n = len(chosenSubSet)
    nPerClass = 50
    nPerObj = 1
    latentCat = input_output.get_latent(chosenSubSet, nPerClass, nPerObj)
    
    tSNE(latentCat, n)
    
    plt.show()


if __name__ == '__main__':
    _main()
