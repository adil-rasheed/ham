'''This file contains functions used in the finite element method,
using lagrange polinomials as the basis function'''
import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
import quadrature
import getplate
from matplotlib import pyplot as plt, cm
from utils import zero, length


def L2(pts, f1, f2 = zero):
    '''returns L2 of f1-f2, (just f1 if f2 is empty)'''
    l=0
    def integrand(x): return (f1(x)-f2(x))**2
    for i in range(len(pts)-1):
        l+= quadrature.quadrature1d(pts[i],pts[i+1],5,integrand)
    return l**0.5

def L2_2d(pts,tri, f1, f2 = zero):
    '''returns L2 of f1-f2, (just f1 if f2 is empty)'''
    assert len(pts) == np.amax(tri)+1
    l=0
    def integrand(x):
        assert len(x) == 2
        return (f1(x)-f2(x))**2
    for triangle in tri:
        l+= quadrature.quadrature2d(pts[triangle[0]],pts[triangle[1]],pts[triangle[2]],4,integrand)
    return l**0.5

class Fem_1d:

    def __init__(self, pts, f, p, u_ex=None):
        self.Np = len(pts) #Np number of grid nodes
        self.pts = pts #triangulation, form [a0,a1,a2,a3,...aNp]
        self.f = f # source function
        self.p = p # degree of test functions'''
        self.u_ex = u_ex # exact function if provided
        assert (self.Np-1)%p==0

    def single_point_solution(self, x):
        pts=self.pts
        p=self.p
        vct = self.u_fem
        # we find the element where x is:
        for i in range(0,len(pts)-p,p): # pts[i] will be start of element
            if pts[i+p] >= x: # pts[i+p] is end of element
                result=0
                for j in range(p+1): # j local index of basis function
                    l = 1 # langrange pol
                    for k in range(p+1): # k
                        if k!=j:
                            l*=(x-pts[i+k])/(pts[i+j]-pts[i+k])
                    result += vct[i+j]*l
                return result
        raise ValueError(f'Function evaluated outside domain, x={x}')

    def solution(self, x):
        if length(x)>0:
            return np.array([self.single_point_solution(x_i) for x_i in x])
        else:
            return self.single_point_solution(x)

    def relative_L2(self, ):
        return L2(self.pts,self.u_ex,self.solution)/L2(self.pts,self.u_ex)

    def set_u_ex(self, u_ex):
        self.u_ex = u_ex


class Poisson(Fem_1d):
    def __init__(self, pts, f, p=1, u_ex=None):
        super().__init__(pts,f,p,u_ex)

        Np=self.Np
        F = np.zeros((Np))
        A = np.zeros((Np,Np))

        for k in range(0,Np-p,p): # elementwise: for element [a_k, a_k+p]
            # We loop through all the contributions to A
            for i in range(p+1):
                for j in range(p+1):
                    def integrand(x):
                        # this integrand is nabla(phi_i)*nabla(phi_j)
                        # formula: from rmnotes pg7
                        nphi = 0 #nabla(phi_i)
                        nphj = 0 #nabla(phi_j)
                        for m in range(p+1):
                            if m!=i:
                                producti=1/(pts[k+i]-pts[k+m])
                                for l in range(p+1):
                                    if l!=i and l!=m:
                                        producti *= (x-pts[k+l])/(pts[k+i]-pts[k+l])
                                nphi += producti
                            if m!=j:
                                productj=1/(pts[k+j]-pts[k+m])
                                for l in range(p+1):
                                    if l!=j and l!=m:
                                        productj *= (x-pts[k+l])/(pts[k+j]-pts[k+l])
                                nphj += productj
                        return nphi*nphj
                    A[k+i,k+j] += quadrature.quadrature1d(pts[k],pts[k+p],p, integrand)
                def integrand(x):
                    # this is the integrand phi_i*f
                    phi = 1
                    for j in range(p+1):
                        if j!=i:
                            phi *= (x-pts[k+j])/(pts[k+i]-pts[k+j])
                    return phi*f(x)
                F[k+i] += quadrature.quadrature1d(pts[k],pts[k+p],5, integrand)
        self.A=A
        self.F=F

    def add_Dirichlet_bdry(self, indices=[0,-1], g=None):
        '''g is only needed if super does not have anu u_ex'''
        assert g!=None or self.u_ex!=None
        if length(indices)==0:
            indices = [indices]
            if g!=None:
                assert length(indices) == length(g)
                g = [g]
        if self.u_ex != None:
            g = [self.u_ex(self.pts[i]) for i in indices]
        for i in range(len(indices)):
            self.A[indices[i],:] = 0 # remove redundant equations
            self.A[indices[i],indices[i]] = -1
            self.F[indices[i]] = g[i]

    def add_Neumann_bdry(self, index,h):
        self.F[index] -= h

    def solve(self, ):
        self.u_fem=np.linalg.solve(-self.A,self.F)


class Heat(Fem_1d):
    def __init__(self, pts, f, p=1, u_ex=None, k=None):
        '''With backwards euler, so on step is (M+kA)u_new = u_old + kf

        Np number of grid points
        pts: triangulation, form [a0,a1,a2,a3,...aNp]
        p: polynomial degree of test functoins
        f: source, form f(x,t)
        u_ex: exact solution. Form: u_ex(x,t=T) (important that T is default time,
                                    time is not provided when calculating L2)
        k: time step lenght. Only needs to be given when usin step() and not solve()
        '''
        super().__init__(pts,f,p,u_ex)
        self.time=0
        self.k = k

    def __discretize(self):
        Np = self.Np
        pts= self.pts
        p = self.p
        F = np.zeros((Np))
        A = np.zeros((Np,Np))
        M = np.zeros((Np,Np))

        for k in range(0,Np-p,p): # elementwise: for element [a_k, a_k+p]
            # We loop through all the contributions to A
            for i in range(p+1):
                for j in range(p+1):

                    def integrand(x): # for a_ij
                        # this integrand is nabla(phi_i)*nabla(phi_j)
                        # formula: from rmnotes pg7
                        nphi = 0 #nabla(phi_i)
                        nphj = 0 #nabla(phi_j)
                        for m in range(p+1):
                            if m!=i:
                                producti=1/(pts[k+i]-pts[k+m])
                                for l in range(p+1):
                                    if l!=i and l!=m:
                                        producti *= (x-pts[k+l])/(pts[k+i]-pts[k+l])
                                nphi += producti
                            if m!=j:
                                productj=1/(pts[k+j]-pts[k+m])
                                for l in range(p+1):
                                    if l!=j and l!=m:
                                        productj *= (x-pts[k+l])/(pts[k+j]-pts[k+l])
                                nphj += productj
                        return nphi*nphj
                    A[k+i,k+j] += quadrature.quadrature1d(pts[k],pts[k+p],p, integrand)

                    def integrand(x): # for m_ij
                        # this integrand is phi_i*phi_j
                        # formula: from rmnotes pg7
                        phi = 1 #phi_i
                        phj = 1 #phi_j
                        for m in range(p+1):
                            if m!=i:
                                phi*=(x-pts[k+m])/(pts[k+i]-pts[k+m])
                            if m!=j:
                                phj*=(x-pts[k+m])/(pts[k+j]-pts[k+m])
                        return phi*phj
                    M[k+i,k+j] += quadrature.quadrature1d(pts[k],pts[k+p],p+1, integrand)

                def integrand(x): # for f_i
                    # this is the integrand phi_i*f
                    phi = 1
                    for j in range(p+1):
                        if j!=i:
                            phi *= (x-pts[k+j])/(pts[k+i]-pts[k+j])
                    return phi*self.f(x=x, t=self.time)
                F[k+i] += quadrature.quadrature1d(pts[k],pts[k+p],5, integrand)

        return M,A,F

    def __add_Dirichlet_bdry(self, indices=[0,-1], g=None):
        '''g can be used instead of u_ex, either when u_ex is unknown or for testing with some other g'''
        assert g!=None or self.u_ex!=None
        if length(indices)==0:
            indices = [indices]
            if g!=None:
                assert length(indices) == length(g)
                g = [g]
        if g == None:
            g = [self.u_ex(self.pts[i], t=self.time) for i in indices]
        else:
            g = [gt(t=self.time) for gt in g]
        for i in range(len(indices)):
            self.MA[indices[i],:] = 0 # remove redundant equations
            self.M[indices[i],:] = 0
            self.MA[indices[i],indices[i]] = self.k
            self.F[indices[i]] = g[i]

    def step(self, g=None, correction = 0):
        '''Do one step of the backward euler scheme'''
        u_prev = self.u_fem
        self.time += self.k
        M,A,F = self.__discretize()
        self.F = F
        self.M = M
        self.MA = M+A*self.k
        self.__add_Dirichlet_bdry(g=g)
        self.u_fem = np.linalg.solve(self.MA, M@u_prev+self.F*self.k+correction) #Solve system


    def solve(self, time_steps, u0=None, g=None, T=1, callback=None):
        '''find u at t=T using backward euler'''
        assert self.time == 0 # this is only supposed to run on unsolved systems
        k = T/time_steps
        self.k=k

        if u0 != None:
            self.u_fem = u0(self.pts)
        else:
            self.u_fem = self.u_ex(self.pts, t=0)
        for t in range(time_steps):
            self.step(g)
            if callback!=None:
                callback(self.time,self.u_fem)


def in_triangle(p1,p2,p3,x):
    A = np.array([p2-p1, p3-p1]).T
    y = np.linalg.solve(A, x-p1)
    return y[0]>=-1e-15 and y[1]>=-1e-15 and y[0]+y[1]<=1+1e-15



class Fem_2d:

    def __init__(self, pts, tri, edge, f, p=1, u_ex=None):
        self.pts = pts # nodal points
        self.tri = tri #triangulation
        self.edge = edge # index of nodal points on the edge
        self.f = f # source function
        self.p = p # degree of test functions'''
        self.u_ex = u_ex # exact function if provided
        assert p==1 # p>1 not implemented (no real plans to do so either)

    def single_point_solution(self, x):
        tri=self.tri
        pts=self.pts
        u = self.u_fem
        # we find the element where x is:
        for i1,i2,i3 in tri:
            p1,p2,p3 = pts[i1],pts[i2],pts[i3]
            # Same as in_triangle(), but keep y
            A = np.array([p2-p1, p3-p1]).T
            x_local = np.linalg.solve(A, x-p1)
            if x_local[0]>=-1e-15 and x_local[1]>=-1e-15 and x_local[0]+x_local[1]<=1+1e-15:
                return u[i1] + x_local[0]*(u[i2]-u[i1]) + x_local[1]*(u[i3]-u[i1])
        raise ValueError(f'Function evaluated outside domain, x={x}')

    def solution(self, x):
        if length(x[0])>0: # not single x and y values
            return np.array([self.solution(x_i) for x_i in x])
        else:
            return self.single_point_solution(x)

    def relative_L2(self):
        return L2_2d(self.pts, self.tri,self.u_ex,self.solution)/L2_2d(self.pts, self.tri,self.u_ex)

    def set_u_ex(self, u_ex):
        self.u_ex = u_ex

    def plot_solution(self):

        N=int(len(self.pts)**0.5-1)*4

        X = np.array([np.linspace(-1,1,N+1)]*(N+1)).T
        Y = np.array([np.linspace(-1,1,N+1)]*(N+1))
        u_fem = self.solution(np.array([X.T,Y.T]).T)
        u_ex = self.u_ex(np.array([X.T,Y.T]).T)

        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1,projection='3d')
        ax2 = fig.add_subplot(1,2,2,projection='3d')
        ax1.plot_surface(X,Y,u_fem, cmap=cm.coolwarm)
        ax2.plot_surface(X,Y,u_ex, cmap=cm.coolwarm)
        #ax.view_init(elev=24, azim=-30)
        #ax.set_xlabel('x')
        #ax.set_ylabel('y')
        #ax.set_zlabel('u')
        plt.tight_layout()
        #plt.savefig(f'../preproject/1d_heat_figures/sols/{sol}.pdf')
        plt.show()

class Heat_2d(Fem_2d):
    def __init__(self, pts, tri, edge, f, p=1, u_ex=None, k=None):
        '''With backwards euler, so on step is (M+kA)u_new = u_old + kf

        pts: nodal points, form [[x1, y1], [x2, y2,] ...]
        tri: triangulation, indices of pts, from [[1,2,3], [0,3,4]... ]
               (where pts[1], pts[2], pts[3] makes up on triangle and so on)
        edge: list of indices of boundary nodes. Form [[0,1], [1,5]...]
               (where [pts[0],pts[1]] is a boundary edge and so on)
        p: polynomial degree of test functoins
        f: source, form f(x,t) where x = [x_1, x_2]
        u_ex: exact solution. Form: u_ex(x,t=T) (important that T is default time,
                                    time is not provided when calculating L2)
        k: time step lenght. Only needs to be given when usin step() and not solve()
        '''
        super().__init__(pts, tri, edge, f, p, u_ex)
        self.time=0
        self.k = k
        if self.k != None:
            self.__discretize()

    def __discretize(self):
        Np = len(self.pts)
        A = np.zeros((Np,Np))
        M = np.zeros((Np,Np))

        for I in self.tri: # elementwise
            p0 = self.pts[I[0]] # points in the triangle
            p1 = self.pts[I[1]]
            p2 = self.pts[I[2]]
            # On the triangle phi_a is a+bx+cy such that phi_i(pj)=d_ij
            # We solve a linear system of equations to find a,b,c.
            m = np.array([
                [1, p0[0],p0[1]],
                [1, p1[0],p1[1]],
                [1, p2[0],p2[1]]
                ])
            phi = [0,0,0]
            phi[0] = np.linalg.solve(m, np.array([1,0,0]))
            phi[1] = np.linalg.solve(m, np.array([0,1,0]))
            phi[2] = np.linalg.solve(m, np.array([0,0,1]))
            # We loop through all the contributions to A
            for i in range(3):
                for j in range(3):
                    # The integrand for a_ij is the constant function nabla phi_i * nabla phi_j
                    def integrand_a(x):
                        return phi[i][1]*phi[j][1] + phi[i][2]*phi[j][2]
                    A[I[i],I[j]] += quadrature.quadrature2d(p0,p1,p2,1, integrand_a)

                    def integrand_m(x): # for m_ij, this integrand is phi_i*phi_j
                        return (phi[i][0] + phi[i][1]*x[0] + phi[i][2]*x[1])*(phi[j][0] + phi[j][1]*x[0] + phi[j][2]*x[1])
                    M[I[i],I[j]] += quadrature.quadrature2d(p0,p1,p2,4, integrand_m)

        self.M = M
        self.A = A
        self.MA = M+A*self.k

        # add Dirichlet bdry
        for line in self.edge:
            for point in line: # this loops through every point twice, but ensures all points are considered
                self.MA[point,:] = 0 # remove redundant equations
                self.M[point,:] = 0
                self.MA[point,point] = self.k
        self.MA = spsp.csr_matrix(self.MA)


    def __make_F(self):
        # self.time must be up to date!
        Np = len(self.pts)
        F = np.zeros((Np))
        for I in self.tri: # elementwise
            p0 = self.pts[I[0]] # points in the triangle
            p1 = self.pts[I[1]]
            p2 = self.pts[I[2]]
            # On the triangle phi_a is a+bx+cy such that phi_i(pj)=d_ij
            # We solve a linear system of equations to find a,b,c.
            m = np.array([
                [1, p0[0],p0[1]],
                [1, p1[0],p1[1]],
                [1, p2[0],p2[1]]
                ])
            phi = [0,0,0]
            phi[0] = np.linalg.solve(m, np.array([1,0,0]))
            phi[1] = np.linalg.solve(m, np.array([0,1,0]))
            phi[2] = np.linalg.solve(m, np.array([0,0,1]))
            # We loop through all the contributions to F
            for i in range(3):
                def integrand_f(x): # for f_i, this is the integrand phi_i*f
                    return (phi[i][0] + phi[i][1]*x[0] + phi[i][2]*x[1])*self.f(x=x, t=self.time)
                F[I[i]] += quadrature.quadrature2d(p0,p1,p2,4, integrand_f)
        self.F = F

        # add Dirichlet bdry
        for line in self.edge:
            for point in line: # this loops through every point twice, but ensures all points are considered
                self.F[point] = self.u_ex(x=self.pts[point], t = self.time)

    def step(self, correction = 0):
        '''Do one step of the backward euler scheme'''
        u_prev = self.u_fem
        self.time += self.k
        self.__make_F()
        self.u_fem = spla.spsolve(self.MA, self.M@u_prev+self.F*self.k+correction) #Solve system
        #self.u_fem = np.linalg.solve(self.MA, M@u_prev+self.F*self.k+correction) #Solve system


    def solve(self, time_steps, u0=None, T=1, callback=None):
        '''find u at t=T using backward euler'''
        assert self.time == 0 # this is only supposed to run on unsolved systems
        k = T/time_steps
        self.k=k
        self.__discretize()

        if u0 != None:
            self.u_fem = u0(x=self.pts)
        else:
            self.u_fem = self.u_ex(x=self.pts, t=0)
        assert self.u_fem.shape == (len(self.pts),)
        for t in range(time_steps):
            self.step()
            if callback!=None:
                callback(self.time,self.u_fem)



class Elasticity_2d():
    def __init__(self, pts, tri, edge, f, nu=0.25, p=1, u_ex=None, w_ex=None, static=False, k=None):
        '''
        pts: nodal points, form [[x1, y1], [x2, y2,] ...]
        tri: triangulation, indices of pts, from [[1,2,3], [0,3,4]... ]
               (where pts[1], pts[2], pts[3] makes up on triangle and so on)
        edge: list of indices of boundary nodes. Form [[0,1], [1,5]...]
               (where [pts[0],pts[1]] is a boundary edge and so on)
        p: polynomial degree of test functoins
        nu: Poisson ration - constant used in stiffness tensor C. 0.25 typical value for many solids.
            Note that the other constant are effectively scaled away by x (E) and t (rho). Adjust F and T to compensate.
        f: source, form f(x,t) where x = [x_1, x_2]. should return [f_1, f_2]
        u_ex: exact solution. Form: u_ex(x,t=T) (important that T is default time,
                                    time is not provided when calculating L2).
                                    u_ex should return [u_1, u_2] i.e. [u,v]
        static: bool, if static elasticity equation is used (rho = 0)
        k: time step lenght. Only needs to be given when usin step() and not solve()
        '''
        self.pts = pts # nodal points
        self.tri = tri #triangulation
        self.edge = edge # index of nodal points on the edge
        self.f = f # source function
        self.p = p # degree of test functions'''
        self.u_ex = u_ex # exact function if provided
        self.w_ex = w_ex # exact time differentiated function if provided
        assert p==1 # p>1 not implemented (no real plans to do so either)

        self.static=static
        self.time=0
        self.k = k
        self.nu=nu
        self.C = np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])/(1-nu**2)
        if self.k != None:
            self.__discretize()

    def single_point_solution(self, x):
        tri=self.tri
        pts=self.pts
        u = self.u_fem
        # we find the element where x is:
        for i1,i2,i3 in tri:
            p1,p2,p3 = pts[i1],pts[i2],pts[i3]
            # Same as in_triangle(), but keep y
            A = np.array([p2-p1, p3-p1]).T
            x_local = np.linalg.solve(A, x-p1)
            if x_local[0]>=-1e-15 and x_local[1]>=-1e-15 and x_local[0]+x_local[1]<=1+1e-15:
                return (
                           u[2*i1] + x_local[0]*(u[2*i2]-u[2*i1]) + x_local[1]*(u[2*i3]-u[2*i1]),
                           u[2*i1+1] + x_local[0]*(u[2*i2+1]-u[2*i1+1]) + x_local[1]*(u[2*i3+1]-u[2*i1+1])
                       )
        raise ValueError(f'Function evaluated outside domain, x={x}')

    def solution(self, x):
        if length(x[0])>0: # not single x and y values
            return np.array([self.solution(x_i) for x_i in x])
        else:
            return self.single_point_solution(x)

    def relative_L2(self):
        return self.L2_2d(self.u_ex,self.solution)/self.L2_2d(self.u_ex)

    def L2_2d(self, f1, f2 = lambda x: [0,0]):
        assert len(self.pts) == np.amax(self.tri)+1
        l=0
        def integrand(x):
            assert len(x) == 2
            assert len(f1(x)) == 2
            assert len(f2(x)) == 2
            return (f1(x)[0]-f2(x)[0])**2 + (f1(x)[1]-f2(x)[1])**2
        for triangle in self.tri:
            l+= quadrature.quadrature2d(self.pts[triangle[0]],self.pts[triangle[1]],self.pts[triangle[2]],4,integrand)
        return l**0.5

    def __discretize(self):
        Np = len(self.pts)
        A = np.zeros((2*Np,2*Np))
        M = np.zeros((2*Np,2*Np))

        for I in self.tri: # elementwise
            p0 = self.pts[I[0]] # points in the triangle
            p1 = self.pts[I[1]]
            p2 = self.pts[I[2]]
            # On the triangle phi_a is a+bx+cy such that phi_i(pj)=d_ij
            # We solve a linear system of equations to find a,b,c.
            m = np.array([
                [1, p0[0],p0[1]],
                [1, p1[0],p1[1]],
                [1, p2[0],p2[1]]
                ])
            phi = [0,0,0]
            phi[0] = np.linalg.solve(m, np.array([1,0,0]))
            phi[1] = np.linalg.solve(m, np.array([0,1,0]))
            phi[2] = np.linalg.solve(m, np.array([0,0,1]))
            # We loop through all the contributions to A
            for i in range(6): # p0_u, p0_v, p1_u, p1_v, p2_u, p2_v (u,v components of u)
                for j in range(6):
                    # The integrand for a_ij is the constant function epsilon(phi_i)^T * epsilon(phi_j)
                    epsilon_i = np.array([((i+1)%2)*phi[i//2][1], (i%2)*phi[i//2][2], (i%2)*phi[i//2][1] + ((i+1)%2)*phi[i//2][2]])
                    epsilon_j = np.array([((j+1)%2)*phi[j//2][1], (j%2)*phi[j//2][2], (j%2)*phi[j//2][1] + ((j+1)%2)*phi[j//2][2]])
                    def integrand_a(x):
                        return epsilon_i.T @ self.C @ epsilon_j
                    A[2*I[i//2] + (i%2),2*I[j//2] + (j%2)] += quadrature.quadrature2d(p0,p1,p2,1, integrand_a)

                    def integrand_m(x): # for m_ij, this integrand is phi_i*phi_j
                        return (phi[i//2][0] + phi[i//2][1]*x[0] + phi[i//2][2]*x[1])*(phi[j//2][0] + phi[j//2][1]*x[0] + phi[j//2][2]*x[1])
                    if i%2 == j%2: # dont mix dimensions
                        M[2*I[i//2] + (i%2),2*I[j//2] + (j%2)] += quadrature.quadrature2d(p0,p1,p2,4, integrand_m)
        self.M = M
        self.A = A
        if self.static:
            self.MA = A
        else:
            self.MA = np.zeros((4*Np,4*Np))
            self.MA[:2*Np,:2*Np] = A 
            self.MA[:2*Np,2*Np:] = M/self.k
            self.MA[2*Np:,:2*Np] = -np.identity(2*Np)/self.k
            self.MA[2*Np:,2*Np:] = np.identity(2*Np)

        # add Dirichlet bdry
        for line in self.edge:
            for point in line: # this loops through every point twice, but ensures all points are considered
                self.MA[2*point,:] = 0 # remove redundant equations
                self.MA[2*point+1,:] = 0
                self.M[2*point,:] = 0
                self.M[2*point+1,:] = 0
                self.MA[2*point,2*point] = 1
                self.MA[2*point+1,2*point+1] = 1
        self.MA = spsp.csr_matrix(self.MA)


    def __make_F(self):
        # self.time must be up to date!
        Np = len(self.pts)
        F = np.zeros((2*Np))
        for I in self.tri: # elementwise
            p0 = self.pts[I[0]] # points in the triangle
            p1 = self.pts[I[1]]
            p2 = self.pts[I[2]]
            # On the triangle phi_a is a+bx+cy such that phi_i(pj)=d_ij
            # We solve a linear system of equations to find a,b,c.
            m = np.array([
                [1, p0[0],p0[1]],
                [1, p1[0],p1[1]],
                [1, p2[0],p2[1]]
                ])
            phi = [0,0,0]
            phi[0] = np.linalg.solve(m, np.array([1,0,0]))
            phi[1] = np.linalg.solve(m, np.array([0,1,0]))
            phi[2] = np.linalg.solve(m, np.array([0,0,1]))
            # We loop through all the contributions to F
            for i in range(6): # p0_u, p0_v, p1_u, p1_v, p2_u, p2_v (u,v components of u)
                def integrand_f(x): # for f_i, this is the integrand phi_i*f
                    F = self.f(x, t=self.time)
                    return (F[0]*((i+1)%2) + F[1]*(i%2)) * phi[i//2] @ np.array([1,x[0],x[1]])
                F[I[i//2]*2 + (i%2)] += quadrature.quadrature2d(p0,p1,p2,4, integrand_f)
        self.F = F
        # add Dirichlet bdry
        for line in self.edge:
            for point in line: # this loops through every point twice, but ensures all points are considered
                self.F[2*point:2*point+2] = self.u_ex(x=self.pts[point], t = self.time)



    def step(self, g=None, correction = 0):
        '''Do one step of the backward euler scheme'''
        u_prev = self.u_fem
        w_prev = self.w_fem
        Np = len(self.pts)
        self.time += self.k
        self.__make_F()
        if self.static:
            self.u_fem = spla.spsolve(self.MA, self.F+correction) #Solve system
        else:
            res = spla.spsolve(self.MA, np.concatenate((self.M@w_prev/self.k+self.F+correction, -u_prev/self.k))) #Solve system
            self.u_fem = res[:2*Np]
            self.w_fem = res[2*Np:]


    def solve(self, time_steps, u0=None, T=1, callback=None):
        '''find u at t=T using backward euler'''
        assert self.time == 0 # this is only supposed to run on unsolved systems
        k = T/time_steps
        self.k=k
        self.__discretize()

        if u0 != None:
            self.u_fem = u0(x=self.pts)
            self.w_fem = w0(x=self.pts)
        else:
            self.u_fem = np.ravel(self.u_ex(x=self.pts, t=0))
            if self.w_ex != None:
                self.w_fem = np.ravel(self.w_ex(x=self.pts, t=0))
            else:
                #print('w not provided, approximating')
                self.w_fem = np.ravel(-self.u_ex(x=self.pts, t=0) + self.u_ex(x=self.pts, t=k/10)) * (10/k) # approximate w numerically


        assert self.u_fem.shape == (2*len(self.pts),)
        for t in range(time_steps):
            self.step()
            if callback!=None:
                callback(self.time,self.u_fem)

    def plot_solution(self):

        N=int(len(self.pts)**0.5-1)*4

        X = np.array([np.linspace(-1,1,N+1)]*(N+1)).T
        Y = np.array([np.linspace(-1,1,N+1)]*(N+1))
        u_fem = self.solution(np.array([X.T,Y.T]).T)
        u_ex = self.u_ex(np.array([X.T,Y.T]).T)

        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1,projection='3d')
        plt.title('u_fem')
        ax2 = fig.add_subplot(2,2,2,projection='3d')
        plt.title('u_exact')
        ax3 = fig.add_subplot(2,2,3,projection='3d')
        plt.title('v_fem')
        ax4 = fig.add_subplot(2,2,4,projection='3d')
        plt.title('v_exact')
        ax1.plot_surface(X,Y,u_fem[:,:,0], cmap=cm.coolwarm)
        ax2.plot_surface(X,Y,u_ex[:,:,0], cmap=cm.coolwarm)
        ax3.plot_surface(X,Y,u_fem[:,:,1], cmap=cm.coolwarm)
        ax4.plot_surface(X,Y,u_ex[:,:,1], cmap=cm.coolwarm)
        #ax.view_init(elev=24, azim=-30)
        #ax.set_xlabel('x')
        #ax.set_ylabel('y')
        #ax.set_zlabel('u')
        plt.tight_layout()
        #plt.savefig(f'../preproject/1d_heat_figures/sols/{sol}.pdf')
        plt.show()


class Disc: # short for disctretizatoin

    def __init__(self, T, time_steps, Ne, equation='heat', p=1, dim=1, xa=0, xb=1, ya=0, yb=1, static=False):
        assert p==1 or dim==1
        self.T=T
        self.time_steps = time_steps
        self.k = T/time_steps
        self.p=p
        self.dim=dim
        self.udim=dim if equation=='elasticity' else 1 # dimensionality of u
        self.equation=equation
        self.static=static # only relevant for elasticity
        self.xa=xa
        self.xb=xb
        self.ya=ya
        self.yb=yb
        
        if dim == 1:
            self.pts = np.linspace(xa,xb,Ne*p+1) # Note Ne is in each dimension (for now at least)
            self.pts_fine = np.linspace(xa,xb,Ne*p*8+1)
            self.edge_ids1 = [0,len(self.pts)-1]
            self.edge_ids2 = self.edge_ids1
            self.pts_line = self.pts_fine
        elif dim == 2:
            self.pts, self.tri, self.edge = getplate.getPlate(Ne+1, xa=xa, xb=xb, ya=ya, yb=yb)
            self.pts_fine, self.tri_fine, self.edge_fine = getplate.getPlate(Ne*4+1, xa=xa, xb=xb, ya=ya, yb=yb)
            self.edge_ids1 = self.edge[:,0] # indices of points in pts that are on the edge
            self.edge_ids2 = self.edge_ids1  # indices in u_fem correspondting to the edge points (differ from edge_ids1 when udim>1)
            if equation == 'elasticity':
                self.edge_ids2 = np.ravel(np.array([2*self.edge[:,0], 2*self.edge[:,0]+1]).T)
            assert (np.sort(self.edge[:,1]) == np.sort(self.edge[:,0])).all() # control that all edge points are in first col of edge
            self.pts_line = np.zeros((Ne*p*8+1, 2))
            self.pts_line[:,0] = np.linspace(xa,xb,Ne*p*8+1)
        else:
            raise ValueError(f'Dimentionality dim={dim} is not implemented')
        #self.Np=len(self.pts) # Number of points
        self.Nv=len(self.pts)*self.udim # Number of values (=len of u_fem)
        self.inner_ids1 = np.setdiff1d(np.arange(len(self.pts)), self.edge_ids1)
        self.inner_ids2 = np.setdiff1d(np.arange(self.Nv), self.edge_ids2)

    def make_model(self, f, u_ex=None, w_ex=None):
        if self.equation == 'heat':
            if self.dim == 1:
                return Heat(self.pts, f, p=self.p, u_ex=u_ex, k=self.k)
            elif self.dim == 2:
                assert self.p == 1
                return Heat_2d(self.pts, self.tri, self.edge, f, p=self.p, u_ex=u_ex, k=self.k)
            raise ValueError(f'Dimentionality dim={self.dim} is not implemented')
        elif self.equation == 'elasticity':
            if self.dim == 2:
                return Elasticity_2d(self.pts, tri=self.tri, edge=self.edge, f=f, p=self.p, u_ex=u_ex, k=self.k, w_ex=w_ex, static=self.static)
            raise ValueError(f'Dimentionality dim={self.dim} is not implemented')
        raise ValueError(f'Equation name={self.equation} is not implemented')

    def format_u(self, u):
        if self.equation == 'heat':
            return u
        elif self.equation == 'elasticity':
            assert u.shape[1] == 2
            return np.ravel(u)
        raise ValueError(f'Equation name={self.equation} is not implemented')
