#copied from project in tma4220

import numpy as np

def quadrature1d(a, b, Nq, g):
    """
        Calculate Gaussian quadrature in 1d over the interval [a,b],
        for the function g using Nq quadrature points.
        a - startpoint
        b - endpoint
        Nq - number of quadrature points must be from the list [1, 2, 3, 4, 5].
        g - function to integrate
    """
    x1=(b-a)/2
    x2=(b+a)/2
    if Nq==1:
        I=g(x2)*2
    elif Nq==2:
        I=g(-x1/np.sqrt(3)+x2)+g(x1/np.sqrt(3)+x2)
    elif Nq==3:
        I=8*g(x2)/9+5*(g(x2+x1*np.sqrt(3/5))+g(x2-x1*np.sqrt(3/5)))/9
    elif Nq==4:
        w1=(18-np.sqrt(30))/36
        w2=(18+np.sqrt(30))/36
        y1=np.sqrt((3+2*np.sqrt(6/5))/7)
        y2=np.sqrt((3-2*np.sqrt(6/5))/7)
        I=w1*g(x2-y1*x1)+w2*g(x2-y2*x1)+w2*g(x2+y2*x1)+w1*g(x2+y1*x1)
    elif Nq==5:
        w1=128/225
        w2=(322+13*np.sqrt(70))/900
        w3=(322-13*np.sqrt(70))/900
        y1=0
        y2=np.sqrt(5-2*np.sqrt(10/7))/3
        y3=np.sqrt(5+2*np.sqrt(10/7))/3
        I=w1*g(x2+y1*x1)+w2*g(x2-y2*x1)+w2*g(x2+y2*x1)+w3*g(x2+y3*x1)+w3*g(x2-y3*x1)
    else:
        return "error"
    return I*(b-a)/2

def quadrature2dline(a, b, Nq, g):
    """
        Takes in two points in R2, and a function g. Then approximates the line integral for a straight
        line between a and b, using Gaussian quadrature with Nq points.
        a - startpoint in R2
        b - endpoint in R2
        Nq - number of quadrature points must be from the list [1, 2, 3, 4].
        g - function to integrate
    """
    mid = [(b[0]+a[0])/2, (b[1]+a[1])/2]                #Coordinates of midpoint between a and b
    dist = np.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)/2     #Distance from midpoint to a or b.
    b2 = [b[0]-mid[0], b[1]-mid[1]]                     #Vector from midpoint to b.
    if Nq==1:
        I=g(mid[0], mid[1])*2
    elif Nq==2:
        I=g(-b2[0]/np.sqrt(3)+mid[0], -b2[1]/np.sqrt(3)+mid[1])+ \
            g(b2[0]/np.sqrt(3)+mid[0], b2[1]/np.sqrt(3)+mid[1])
    elif Nq==3:
        I=8*g(mid[0], mid[1])/9+5*(g(mid[0]+b2[0]*np.sqrt(3/5),mid[1]+b2[1]*np.sqrt(3/5))+ \
            g(mid[0]-b2[0]*np.sqrt(3/5),mid[1]-b2[1]*np.sqrt(3/5)))/9

    elif Nq==4:
        w1=(18-np.sqrt(30))/36
        w2=(18+np.sqrt(30))/36
        y1=np.sqrt((3+2*np.sqrt(6/5))/7)
        y2=np.sqrt((3-2*np.sqrt(6/5))/7)
        I=w1*g(mid[0]-y1*b2[0], mid[1]-y1*b2[1])+\
            w2*g(mid[0]-y2*b2[0], mid[1]-y2*b2[1])+ \
            w2*g(mid[0]+y2*b2[0], mid[1]+y2*b2[1])+ \
            w1*g(mid[0]+y1*b2[0], mid[1]+y1*b2[1])
    else:
        return "error"
    return I*dist

def quadrature2d(p1, p2, p3, Nq, g_vect):
    """
        Calculate Gaussian quadrature over a triangle with corners p1, p2 and p3 for function g.
        p1 - First vertex of triangle
        p2 - Second vertex of triangle
        p3 - third vertex of triangle
        Nq - number of quadrature points
        g - pointer to function to integrate
    """
    g = lambda x,y: g_vect([x,y]) # workaround for different syntax convetions in these files. TODO: change this file to vector format

    #The matrix A stores all information about the transform to the reference element
    A = [p2[0]-p1[0], p3[0]-p1[0], p2[1]-p1[1], p3[1]-p1[1]]

    if Nq == 1:
        res = g((A[0]+A[1])/3+p1[0], (A[2]+A[3])/3+p1[1])
    elif Nq == 3:
        res = g(0.5*(A[0]+A[1])+p1[0], 0.5*(A[2]+A[3])+p1[1])/3
        res = res + g(0.5*(A[0])+p1[0], 0.5*(A[2])+p1[1])/3
        res = res + g(0.5*(A[1])+p1[0], 0.5*(A[3])+p1[1])/3
    elif Nq ==4:
        res = (25/48)*g(0.2*(A[0]+A[1])+p1[0], 0.2*(A[2]+A[3])+p1[1])
        res = res + (25/48)*g(0.2*A[0]+0.6*A[1]+p1[0], 0.2*A[2]+0.6*A[3]+p1[1])
        res = res + (25/48)*g(0.6*A[0]+0.2*A[1]+p1[0], 0.6*A[2]+0.2*A[3]+p1[1])
        res = res - (9/16)*g((A[0]+A[1])/3+p1[0], (A[2]+A[3])/3+p1[1])

    return res*(np.abs(A[1]*A[2]-A[0]*A[3]))/2          #Divide by 2 as sum of weights should be 1/2 for the reference element.

if __name__=="__main__":
    print("Testing for 1d-quadrature with 1d integral: \n")
    f = lambda x: np.exp(x)
    I = np.exp(2)-np.exp(1)                                             #Exact value of integral
    for i in range(5):
        print("Testing with ", i+1, "quadrature points gives:")
        I_num = quadrature1d(1, 2, i+1, f)
        print("Integral value equal to: ", I_num)
        print("Absolute error equal to: ", np.abs(I-I_num), '\n')

    print("Testing polynomials with 1d-quadrature: \n")
    for i in range(1,5):
        f1 = lambda x: x**(2*i+1)
        f2 = lambda x: x**(2*i+2)
        I1 = 1/(2*i+2)
        I2 = 1/(2*i+3)
        print("Testing with ", i+1, "quadrature points gives:")
        I1_num = quadrature1d(0, 1, i+1, f1)
        I2_num = quadrature1d(0, 1, i+1, f2)
        I3_num = quadrature1d(0, 1, i, f1)
        print("First integral (should be exact):")
        print("Integral value equal to: ", I1_num)
        print("Absolute error equal to: ", np.abs(I1-I1_num), '\n')

        print("Second integral (should not be exact):")
        print("Integral value equal to: ", I2_num)
        print("Absolute error equal to: ", np.abs(I2-I2_num), '\n')

        print("First integral with less points (should not be exact):")
        print("Integral value equal to: ", I1_num)
        print("Absolute error equal to: ", np.abs(I1-I3_num), '\n')

    quit()

    print("Testing for 1d-quadrature with line integral method: \n")
    h = lambda x,y: np.exp(x)
    a = [1, 0]
    b = [2, 0]
    for i in range(4):
        print("Testing with ", i+1, "quadrature points gives:")
        I_num = quadrature2dline(a, b, i+1, h)
        print("Integral value equal to: ", I_num)
        print("Absolute error equal to: ", np.abs(I-I_num), '\n')

    print("Testing for 2d-quadrature with points going clockwise: \n")
    g = lambda x,y: np.log(x+y)
    a = [1,0]
    c = [3,1]
    b = [3,2]
    I = 1.16542                                                         #Exact value of integral
    Nq = [1, 3, 4]
    for i in range(3):
        print("Testing with ", Nq[i], "quadrature points gives:")
        I_num = quadrature2d(a, b, c, Nq[i], g)
        print("Integral value equal to: ", I_num)
        print("Absolute error equal to: ", np.abs(I-I_num), '\n')

    print("Testing for 2d-quadrature with points going counter clockwise: \n")
    g = lambda x,y: np.log(x+y)
    a = [1,0]
    b = [3,1]
    c = [3,2]
    I = 1.16542                                                         #Exact value of integral
    Nq = [1, 3, 4]
    for i in range(3):
        print("Testing with ", Nq[i], "quadrature points gives:")
        I_num = quadrature2d(a, b, c, Nq[i], g)
        print("Integral value equal to: ", I_num)
        print("Absolute error equal to: ", np.abs(I-I_num), '\n')


    if mode == 'bugfix':
        Ne = 5
        time_steps = 20
        DNNkwargs = {'n_layers':6,'depth':20,'lr':5e-3,'patience':[10,20], 'epochs':[100,100], 'min_epochs':[50,50]}
        pgDNNkwargs = {'n_layers_1':4,'n_layers_2':2,'max_depth':20,'min_depth':8,'lr':5e-3,'patience':[10,20], 'epochs':[100,100], 'min_epochs':[50,50]}#, 'l1_penalty':0.01}
        LSTMkwargs = {'lstm_layers':2, 'lstm_depth':20, 'dense_layers':1, 'dense_depth':20, 'lr':5e-3, 'patience':[10,10], 'epochs':[100,100], 'min_epochs':[50,50]}
        NoM = 2
        time_delta = 5
    elif mode == 'quick_test':
        Ne = 20
        time_steps = 500
        DNNkwargs = {'n_layers':6,'depth':80, 'lr':8e-5, 'patience':[20,20]}
        pgDNNkwargs = {'n_layers_1':3,'n_layers_2':4,'max_depth':125,'min_depth':5,'lr':8e-5,'patience':[20,20]}
        LSTMkwargs = {'lstm_layers':4, 'lstm_depth':80, 'dense_layers':2, 'dense_depth':80, 'lr':8e-5, 'patience':[20,20]}
        NoM=3
        time_delta = 0.3 # max 30 steps back
    elif mode == 'full_test':
        Ne = 20
        time_steps = 5000
        DNNkwargs = {'n_layers':6,'depth':80, 'lr':1e-5, 'patience':[20,20]}
        pgDNNkwargs = {'n_layers_1':3,'n_layers_2':4,'max_depth':125,'min_depth':5,'lr':1e-5,'patience':[20,20]}
        LSTMkwargs = {'lstm_layers':4, 'lstm_depth':80, 'dense_layers':2, 'dense_depth':80, 'lr':1e-5, 'patience':[20,20]}
        NoM=10
        time_delta = 0
