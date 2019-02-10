import torch
import tSNE
import numpy as np


class simpleClassifier(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(simpleClassifier, self).__init__()
        self.linear = torch.nn.Linear(D_in, D_out)
        #self.linear = torch.nn.Sequential(torch.nn.Linear(D_in, D_out, bias=True),)
        #self.logprob = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.linear(x)

        #x = self.linear(x)
        #return x
    
    def fit(self, list_x, list_y, epochs, t_eval, test_x, test_y):
        """x, y : tensors holdind inputs and outputs"""
        
        list_y = [torch.tensor([y]) for y in list_y]
        
        # Construct our loss function and an Optimizer. The call to model.parameters()
        # in the SGD constructor will contain the learnable parameters of the
        # modules which are members of the model.
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        precisions = []
        
        for t in range(epochs):
            s = 0
            for x, y in zip(list_x, list_y):
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = self.forward(x.data)
            
                # Compute and print loss
                loss = criterion(y_pred.reshape(1,-1), y)
                s += loss.item()
            
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if t%t_eval==0:
                self.eval()

                #précision sur test
                pred_test = self.predict(test_x)
                prec = sum(np.array(pred_test) == test_y) / len(test_y)
                precisions.append(prec)

                print("time", t,
                      "precision test", round(prec,2),
                      "loss %.0e" % np.exp(s/len(list_x)))
                self.train()
                
                if np.exp(s/len(list_x))<1e-50:
                    #convergence finie, ca ne sert à rien de continuer
                    break

        #apprentissage fini, passage en mode évaluation
        self.eval()
    
    def predict(self, list_x):
        res = []
        for x in list_x:
            y = self(x)
            res.append(max([0,1], key=lambda i: y[i]))
        return res


def main():
    #chosenSubSet = [0,2] #points rouge et noir qui sont bien DISTINCTS sur la visualisation tSNE
    chosenSubSet = [2, 7] #verts et gris qui sont CONFONDUS sur notre INSTANCE du tSNE
    nCat = len(chosenSubSet)
    nPerCat = 100
    nPerObj = 1
    ratio_train = .5
    
    latentVectors = tSNE.get_latent(chosenSubSet, nPerCat, nPerObj)
    
    labels = ([0] * (len(latentVectors)//2)) + ([1] * (len(latentVectors)//2))
    
    import random
    indexes = list(range(len(latentVectors)))
    random.shuffle(indexes)
    n_train = int(ratio_train * len(indexes))
    train_x = [latentVectors[i] for i in indexes[:n_train]]
    test_x = [latentVectors[i] for i in indexes[n_train:]]
    
    train_y = [labels[i] for i in indexes[:n_train]]
    test_y = [labels[i] for i in indexes[n_train:]]
    
    #construction de notre modèle
    model = simpleClassifier(len(latentVectors[0]), nCat)
    
    #apprentissage
    epochs = 1000 #aura convergé avant
    model.fit(train_x, train_y, epochs, 1, test_x, np.array(test_y))
    
    #précision sur train
    pred_train = model.predict(train_x)
    print("précision train ", sum(np.array(pred_train) == np.array(train_y)) / len(train_y))
    
    #précision sur test
    pred_test = model.predict(test_x)
    print("précision test ", sum(np.array(pred_test) == np.array(test_y)) / len(test_y))
    

if __name__ == '__main__':
    main()
