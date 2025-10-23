import numpy as np

w = np.random.randn(10,3072)*0.001
best_loss = float('inf')
L = lambda x,y,w:np.sum((x.dot(w.T)-y)**2)
xTr_cols = np.random.randn(500,3072)
yTr = np.random.randn(500,10)
for i in range(1000000):
    step_size = 0.0001
    Wtry = w + np.random.randn(10,3072)*step_size
    loss = L(xTr_cols,yTr,Wtry)
    if(loss<best_loss):
        w = Wtry
        best_loss = loss
        print('iter %d loss is %f' %(i,best_loss))
