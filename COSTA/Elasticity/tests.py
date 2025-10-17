# This file contains tests to verify that the code in this project works as it should
import numpy as np
import quadrature
import functions
import FEM
import solvers
from matplotlib import pyplot as plt, cm
import getplate
import sympy as sp
import utils

def test_2d_elast():
    static=True
    T=1
    for i in [0,1,2]:
        f,u,w = functions.manufacture_elasticity_solution(d1=2, d2=2, static=static, **functions.ELsols[i])
        sol = functions.Solution(T=T, f_raw=f, u_raw=u, zero_source=False, name=f'ELsol{i}')
        sol.set_alpha(1)
        print('solving with increasing resolution, error (printed below) should approach 0')
        for n in [3, 5, 7, 9, 11,13,15]:
            pts, tri, edge = getplate.getPlate(n)
            model = FEM.Elasticity_2d(pts, tri, edge, sol.f, u_ex=sol.u, static=static)
            model.solve(time_steps=n, T=sol.T)
            print(model.relative_L2())
        model.plot_solution()

    #nu=0.5
    #def u_ex(x, t=1):
    #    if len(np.array(x).shape)>1:
    #        return (np.array([u_ex(xi,t) for xi in x]))
    #    res = (x[0]**2-1)*(x[1]**2-1)
    #    return [res,res]
    #def f(x,t):
    #    if len(np.array(x).shape)>1:
    #        return (np.array([f(xi,t) for xi in x]))
    #    f1 = 1/(1-nu**2) * (-2*x[1]**2-x[0]**2+nu*x[0]**2-2*nu*x[0]*x[1]-2*x[0]*x[1]+3-nu)
    #    f2 = 1/(1-nu**2) * (-2*x[0]**2-x[1]**2+nu*x[1]**2-2*nu*x[0]*x[1]-2*x[0]*x[1]+3-nu)
    #    return [f1,f2]
    #
    #for n in [3, 5, 7, 9, 11,13, 15, 17, 20]:
    #    pts, tri, edge = getplate.getPlate(n)
    #    model = FEM.Elasticity_2d(pts, tri, edge, f, nu, u_ex=u_ex, static=True)
    #    model.solve(time_steps=1, T=1)
    #    print(model.relative_L2())
    #model.plot_solution()

def test_manufacture_heat_var_k():
    t = sp.symbols('t')
    x = sp.symbols('x')
    alpha = sp.symbols('alpha')

    u = t*x*x
    f,u = functions.manufacture_solution(u,t,[x], k=alpha*u, alpha_var=alpha, d1=1,d2=1)

    u = sp.sin(alpha*t+x)
    f,u = functions.manufacture_solution(u,t,[x], k=u*u, alpha_var=alpha, d1=1,d2=1)

    u = t*x*x
    f,u = functions.manufacture_solution(u,t,[x], k=alpha*u/10+1, alpha_var=alpha, d1=1,d2=1)
    
    u = 2+alpha*(x-1)*sp.tanh(x/(t+0.1))
    f,u = functions.manufacture_solution(u,t,[x], k=sp.sin(4*u)/2 + 1, alpha_var=alpha, d1=1,d2=1)
    
    u = sp.cos(x)*(sp.exp(-t/5)+sp.exp((t-1)/5)*2)
    f,u = functions.manufacture_solution(u,t,[x], k=alpha*u/10+1, alpha_var=alpha, d1=1,d2=1)
    quit()

def test_2d_heat():
    t = sp.symbols('t')
    x = sp.symbols('x')
    y = sp.symbols('y')
    alpha = sp.symbols('alpha')
    #u = sp.exp(-t/(3.5-x**2-(y-(1+alpha)/4)**2))
    u = 2*x*x+3*y*y+t

    f,u = functions.manufacture_solution(u,t,[x,y],alpha_var=alpha, d1=2,d2=2)
    sol = functions.Solution(T=1, f_raw=f, u_raw=u, zero_source=False, name=f'2d_tst1')
    sol.set_alpha(1)

    print('The following 6 numbers are l2 for increasing amount of nodes and steps, so they should approach zero:')
    n=3
    pts, tri, edge = getplate.getPlate(n, 0,1,0,1)
    fem_model = FEM.Heat_2d(pts, tri, edge, f=sol.f, p=1, u_ex=sol.u)
    fem_model.solve(5)
    print(np.sqrt(np.mean((fem_model.u_fem - sol.u(pts))**2)))
    #print(fem_model.relative_L2())
    #fem_model.plot_solution()
    n=6
    pts, tri, edge = getplate.getPlate(n, 0,1,0,1)
    fem_model = FEM.Heat_2d(pts, tri, edge, f=sol.f, p=1, u_ex=sol.u)
    fem_model.solve(20)
    print(np.sqrt(np.mean((fem_model.u_fem - sol.u(pts))**2)))
    #print(fem_model.relative_L2())
    #fem_model.plot_solution()
    n=10
    pts, tri, edge = getplate.getPlate(n, 0,1,0,1)
    fem_model = FEM.Heat_2d(pts, tri, edge, f=sol.f, p=1, u_ex=sol.u)
    fem_model.solve(50)
    print(np.sqrt(np.mean((fem_model.u_fem - sol.u(pts))**2)))
    #print(fem_model.relative_L2())
    #fem_model.plot_solution()
    n=14
    pts, tri, edge = getplate.getPlate(n, 0,1,0,1)
    fem_model = FEM.Heat_2d(pts, tri, edge, f=sol.f, p=1, u_ex=sol.u)
    fem_model.solve(100)
    print(np.sqrt(np.mean((fem_model.u_fem - sol.u(pts))**2)))
    #print(fem_model.relative_L2())
    #fem_model.plot_solution()
    n=17
    pts, tri, edge = getplate.getPlate(n, 0,1,0,1)
    fem_model = FEM.Heat_2d(pts, tri, edge, f=sol.f, p=1, u_ex=sol.u)
    fem_model.solve(200)
    print(np.sqrt(np.mean((fem_model.u_fem - sol.u(pts))**2)))
    #print(fem_model.relative_L2())
    #fem_model.plot_solution()
    n=20
    pts, tri, edge = getplate.getPlate(n, 0,1,0,1)
    fem_model = FEM.Heat_2d(pts, tri, edge, f=sol.f, p=1, u_ex=sol.u)
    fem_model.solve(500)
    print(np.sqrt(np.mean((fem_model.u_fem - sol.u(pts))**2)))
    #print(fem_model.relative_L2())
    #fem_model.plot_solution()



def test_fem_2d():
    n = 8
    pts, tri, edge = getplate.getPlate(n)
    vct = np.zeros(len(pts))
    vct[33]=1
    vct[34]=1
    vct[35]=1
    vct[36]=1
    vct[37]=1
    vct[38]=1
    vct[39]=1
    vct[43]=1
    vct[44]=1
    vct[45]=1
    vct[46]=1
    vct[47]=1
    vct[48]=1
    vct[49]=1
    vct[50]=1
    vct[53]=1
    vct[52]=1
    vct[51]=1

    fem_model = FEM.Fem_2d(pts, tri, edge, f=FEM.zero, p=1, u_ex=None)
    fem_model.u_fem = vct
    fem_model.u_ex = lambda x : np.ones(np.array(x).shape[:-1])
    print(fem_model.relative_L2())
    fem_model.plot_solution()


def test_in_triangle():
    print('--- INTERPRETING THE TRIANGULATION PLOT ---')
    print('Testing the FEM.in_triangle functions ability to find the correct triangle')
    print('Red circle is a random point, and the triangle it is in should have red nodes')
    print('Green circle is on a random line, and all the (up to two) triangles sharing it should have green nodes')
    print('Yellow circle is random node, all triangles around it should have yellow nodes')
    print('Note that overlapping may occur, making it seem incorrect. Yellow is plotted first, then green then red.')
    # make triangulation
    N = 12
    p, tri, edge = getplate.getPlate(N)
    # print triangulation
    for triangle in tri:
        for local_line in [[0,1],[1,2],[2,0]]: # Plot edges of each triangle
            line = [triangle[local_line[0]], triangle[local_line[1]]]
            plt.plot([p[line[0],0], p[line[1],0]],[p[line[0],1], p[line[1],1]],'k-', linewidth=0.5)
    for line in edge: # Plot thicker edges along the boundary
        plt.plot([p[line[0],0], p[line[1],0]],[p[line[0],1], p[line[1],1]],'k-')

    plt.plot(p[:,0], p[:,1], 'bx') # Plot nodes
    # Pick random nodal point, find triangles, and plot
    x = p[np.random.randint(N**2)]
    for t in tri:
        if FEM.in_triangle(p[t[0]],p[t[1]],p[t[2]],x):
            plt.plot(p[t[:],0],p[t[:],1], f'yx') # Plot nodes
    plt.plot(x[0],x[1], f'yo')
    # Pick random point on a line, find triangles, and plot
    chosen_triangle = tri[np.random.randint(len(tri))]
    chosen_pt = np.random.randint(3)
    x = p[chosen_triangle[chosen_pt]] + (p[chosen_triangle[(chosen_pt+1)%3]]-p[chosen_triangle[chosen_pt]]) * np.random.random(1)
    for t in tri:
        if FEM.in_triangle(p[t[0]],p[t[1]],p[t[2]],x):
            plt.plot(p[t[:],0],p[t[:],1], f'gx') # Plot nodes
    plt.plot(x[0],x[1], f'go')
    # Pick random point, find triangle(s), and plot
    x = np.random.random(2)*2 - np.array([1,1])
    for t in tri:
        if FEM.in_triangle(p[t[0]],p[t[1]],p[t[2]],x):
            plt.plot(p[t[:],0],p[t[:],1], f'rx') # Plot nodes
    plt.plot(x[0],x[1], f'ro')
    plt.show()

def test_reshaping():
    # Tests solvers.merge_first_dims
    i=3
    j=5
    k=7
    a = np.arange(i*j*k)
    b = np.reshape(a, (i,j,k))
    c = solvers.merge_first_dims(b)
    for ii in range(i):
        for jj in range(j):
            for kk in range(k):
                assert c[jj*i+ii,kk] == b[ii,jj,kk]
    print('reshape function passed')

def test1():
    def f(x): return 2*(1+x)**-3
    def u_ex(x): return 1/(1+x)-1+x/2
    e_prev=10
    for p in [1,2,3,4]:
        e=test_function(u_ex, f, p)
        # havent thought about the convergence rate here,
        # so just check if -> 0
        assert e < e_prev/2

    def f(x): return 6*x + np.pi**2/4*np.sin(np.pi*x/2)
    def u_ex(x): return x**3 - np.sin(np.pi/2*x)
    e_prev=10
    for p in [1,2,3,4]:
        e=test_function(u_ex, f, p)
        assert e < e_prev/2

    def u_ex(x): return 1/(1+x)-1+x/2
    def f(x): return 2*(1+x)**-3
    for p in [1,2,3,4,5]:
        e_prev=10
        for Ne in [1,2,3,4,5]:
            Np=p*Ne+1
            e=test_function(u_ex, f, p, Np, True)
            assert e < e_prev/2
    print('test1 passed (FEM.Poisson, error -> 0 for increased p and Np, for dirichlet and neumann bdry cond)')

def test_function(u_ex, f, p, Np=13, neumann=False):
    model = FEM.Poisson(np.linspace(0,1,Np),f,p,u_ex)
    if not neumann:
        model.add_Dirichlet_bdry([0,-1])
    else:
        model.add_Dirichlet_bdry([0])
        model.add_Neumann_bdry([-1], 1/4)
    model.solve()
    return model.relative_L2()



def abdullah_bug_test():
    '''Taken from abdullahs preproject thesis chp 8, remember to credit'''
    '''Not implemented completely yet, only have linear elements so far'''#TODO
    fig1, axs1 = plt.subplots(3,2)
    fig2, axs2 = plt.subplots(3,2)
    for exponent in range(1,7):
        def u_ex(x): return x**exponent
        def f(x):
            if exponent <2:
                return 0
            return exponent*(exponent-1)*x**(exponent-2)

        Nes = [2,4,8,16,32]#,64]#, 128, 256]
        ps = [1,2,3,4,5]
        L2s= np.zeros((len(Nes), len(ps)))
        for i in range(len(Nes)):
            for j in range(len(ps)):
                Ne=Nes[i]
                p=ps[j]
                Np = Ne*p+1

                # Discretization:
                model = FEM.Poisson(np.linspace(0,1,Np),f,p,u_ex)
                model.add_Dirichlet_bdry()
                model.solve ()
                L2s[i,j] = model.relative_L2()

                ##pts_fine = np.linspace(0,1,(Np-1)*10+1)
                ##u_fem_fnc = FEM.fnc_from_vct(pts,u_fem,p)
                ##plt.plot(pts_fine, u_ex(pts_fine), label='u_ex')
                ##plt.plot(pts_fine, u_fem_fnc(pts_fine), label='fem')
                ##plt.legend()
                ##plt.show()

        ##plt.loglog(Nes,L2s, label=ps)
        ##plt.xlabel('Number of elements')
        ##plt.grid()
        ##plt.legend(title='p:')
        ##plt.show()
        m = (exponent-1)//2
        n = (exponent-1)%2
        axs1[m,n].loglog(Nes,L2s, label=ps)
        axs1[m,n].set_xlabel('Number of elements')
        axs1[m,n].grid()
        axs1[m,n].legend(title='p:')

        ##plt.plot(ps,L2s.T, label=Nes)
        ##plt.yscale('log')
        ##plt.xlabel('Polynomial degree')
        ##plt.grid()
        ##plt.legend(title='Ne:')
        ##plt.show()
        axs2[m,n].plot(ps,L2s.T, label=Nes)
        axs2[m,n].set_yscale('log')
        axs2[m,n].set_xlabel('Polynomial degree')
        axs2[m,n].grid()
        axs2[m,n].legend(title='N:')
    plt.show()


def sindres_mfact_test(sol=0, alpha=0.5, p=4):
    '''testing equations from sindres paper.'''
    T=1
    solution = functions.Solution(T, f_raw=functions.SBMFACT[sol][0], u_raw=functions.SBMFACT[sol][1], zero_source=False, name=f'{sol}')
    solution.set_alpha(alpha)


    Ne=1
    time_steps=1
    L2s=[]
    dofs=[]
    for i in range(5):
        pts = np.linspace(0,1,Ne*p+1)
        model = FEM.Heat(pts, solution.f, p, solution.u)
        model.solve(time_steps, T=T)
        u_fem = model.u_fem
        pts_fine = np.linspace(0,1,10*Ne*p+1)

        plt.plot(pts_fine,model.solution(pts_fine), label=f'Ne,ts={Ne},{time_steps}')
        dofs.append((Ne*p-1)*time_steps)
        Ne*=2
        time_steps*=2
        L2s.append(model.relative_L2())
    plt.plot(pts_fine, solution.u(pts_fine), label='u_ex')
    plt.legend(title=f'sol {solution.name}')
    plt.show()

    plt.loglog(dofs,L2s)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    test_2d_heat()
    quit()
    test_manufacture_heat_var_k()
    test_2d_elast()
    test_fem_2d()
    test_in_triangle()
    test_reshaping()
    test1()
    #abdullah_bug_test()
    sindres_mfact_test(0,1)
    sindres_mfact_test(1,4)
    sindres_mfact_test(2,3)
    sindres_mfact_test(3,2)
    sindres_mfact_test(4,1)
    sindres_mfact_test(4,2)
    quit()
