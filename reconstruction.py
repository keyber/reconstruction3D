import torch
import tSNE
import numpy as np
import ply

def chamfer(F, x, P, S):
    """return chamfer loss between
    the generated set G = {f(x, p)  for each f, x, p}
    and the real set S"""
    
    loss = 0
    #pour chaque p et f la distance au s le plus proche
    #les MLP doivent bien atteindre un Q
    for p in P:
        for f in F:
            loss += min((f[x][p] - s) ** 2 for s in S)
    
    #pour chaque s la distance au p, MLP le plus proche
    #les Q doivent bien être atteint par un MLP
    for s in S:
        loss += min(min((f[x][p] - s) ** 2
                        for p in P) for f in F)
    return loss
    

class simpleReconstructor(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()
        self.linear = torch.nn.Linear(D_in, D_out)
    
    def forward(self, x):
        """In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors"""
        return self.linear(x)
    
    def fit(self, list_x, list_y, epochs, t_eval, test_x, test_y):
        """x, y : tensors holdind inputs and outputs"""
        
        list_y = [torch.tensor([y]) for y in list_y]
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        precisions = []
        
        for t in range(epochs):
            s = 0
            for x, y in zip(list_x, list_y):
                y_pred = self.forward(x.data)
                
                loss = chamfer(y_pred.reshape(1, -1), y)
                s += loss.item()
                
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if t % t_eval == 0:
                self.eval()
                
                #précision sur test
                pred_test = self.predict(test_x)
                prec = sum(np.array(pred_test) == test_y) / len(test_y)
                precisions.append(prec)
                
                print("time", t,
                      "precision test", round(prec, 2),
                      "loss %.0e" % np.exp(s / len(list_x)))
                self.train()
                
                if np.exp(s / len(list_x)) < 1e-50:
                    #convergence finie, ca ne sert à rien de continuer
                    break
        
        #apprentissage fini, passage en mode évaluation
        self.eval()
    
    def predict(self, list_x):
        res = []
        for x in list_x:
            y = self(x)
            res.append(max([0, 1], key=lambda i: y[i]))
        return res


def _main():
    path = "../AtlasNet/data/ShapeNetCorev2Normalized/02691156_normalised/d1a8e79eebf4a0b1579c3d4943e463ef.ply"
    a = ply.read_ply(path)
    print(a)
    path = "../AtlasNet/data/customShapeNet/02933112/ply/1a1b62a38b2584874c62bee40dcdc539.points.ply"
    a = ply.read_ply(path)
    print(a)

if __name__ == '__main__':
    _main()
