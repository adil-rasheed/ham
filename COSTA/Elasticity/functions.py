import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
from utils import length


# Sindre Blakseth's manufactured solutions:
SBMFACT = [
            (
                lambda x,t,alpha,time_delta: 0,
                lambda x,t,alpha,time_delta: alpha*(t+x*x/2)
            ),
            (
                lambda x,t,alpha,time_delta: 1-alpha,
                lambda x,t,alpha,time_delta: t+alpha*x*x/2
            ),
            (
                lambda x,t,alpha,time_delta: 0.5/(t+alpha+1+time_delta)**0.5-120*x*x-60*x+40,
                lambda x,t,alpha,time_delta: (t+alpha+1+time_delta)**0.5 + 10*x*x*(x-1)*(x+2)
            ),
            (
                lambda x,t,alpha,time_delta: alpha/(t+0.1+time_delta)**2*(x*(1-x)+2*((x-1)*np.tanh(x/(t+0.1+time_delta))-t-0.1+time_delta))*np.cosh(x/(t+0.1+time_delta))**-2,
                lambda x,t,alpha,time_delta: 2+alpha*(x-1)*np.tanh(x/(t+0.1+time_delta))
            ),
            (
                lambda x,t,alpha,time_delta: 2*np.pi*(np.cos(2*np.pi*t+alpha)+2*np.pi*np.sin(2*np.pi*t+alpha))*np.cos(2*np.pi*x) ,
                lambda x,t,alpha,time_delta: 1+np.sin(2*np.pi*t+alpha)*np.cos(2*np.pi*x)
            ),
    ]

# For tuning:
SBMFACT_TUNING = [
            (
                lambda x,t,alpha,time_delta: 0.5/(t+alpha+1+time_delta)**0.5-84*x*x-42*x+28,
                lambda x,t,alpha,time_delta: (t+alpha+1+time_delta)**0.5 + 7*x*x*(x-1)*(x+2)
            ),
            (
                lambda x,t,alpha,time_delta: x**3*(x-alpha)/(t+0.1+time_delta)**2 + (12*x**2-6*alpha*x)/(t+0.1+time_delta),
                lambda x,t,alpha,time_delta: -x**3*(x-alpha)/(t+0.1+time_delta)
            ),
    ]


class Solution:
    def __init__(self, T, f_raw, u_raw, zero_source=True, name='?', time_delta=0, w_raw=None):
        self.T = T # final time (default in u)
        self.f_raw = f_raw # raw versions of function are function of (x,t,T,alpha,time_delta)
        self.u_raw = u_raw
        self.w_raw = w_raw
        self.zero_source = zero_source
        self.name=name # for plot legends
        self.time_delta = time_delta # time_delta - shift in time, required for using negative times on coarse time grids.

    def set_alpha(self, alpha):
        self.alpha = alpha

    def u(self, x, t=None):
        if t is None:
            t = self.T
        return self.u_raw(x=x, t=t, alpha=self.alpha, time_delta=self.time_delta)

    def f(self, x, t=None):
        if t is None:
            t = self.T
        result = np.array(self.f_raw(x=x, t=t, alpha=self.alpha, time_delta=self.time_delta))
        if self.zero_source:
            result *= 0
        return result

    def w(self, x, t=None):
        if t is None:
            t = self.T
        return self.w_raw(x=x, t=t, alpha=self.alpha, time_delta=self.time_delta)

def manufacture_solution(u, t_var, x_vars, k=sp.Integer(1), alpha_var=None, d1=2, d2=1):
    '''Manufactures f from the 2d or 3d heat equation given u (sympy equation)
    returns functions with unused dimensions =0 (eg z=0 for 3d-2d)'''
    #print(d1,d2)
    assert d1>=d2
    assert d1<4
    assert d2>0
    f = u.diff(t_var)
    for d in range(d1):
        f -= u.diff(x_vars[d], x_vars[d])*k
        f -= u.diff(x_vars[d])*k.diff(x_vars[d])
    #print('f:', f)
    f_temp = lambdify([*[x_vars],t_var,alpha_var],f, "numpy")
    u_temp = lambdify([*[x_vars],t_var,alpha_var],u, "numpy")



    # For controlling and/or plotting level of nonlinearity
    k_temp = lambdify([*[x_vars],t_var,alpha_var],k, "numpy")
    global k_max, k_min, u_max, u_min
    k_max = -100
    k_min = 100
    u_max = -100
    u_min = 100
    
    # make functions
    def f(x,t,alpha,time_delta=0):
        x = [x] if length(x) == 0 else x
        k_val = k_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha) 
        u_val = u_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha) 
        global k_max, k_min, u_max, u_min
        if k_val < k_min:
            k_min = k_val
        if k_val > k_max:
            k_max = k_val
        if u_val < u_min:
            u_min = u_val
        if u_val > u_max:
            u_max = u_val
        return f_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha)
    def u(x,t,alpha,time_delta=0):
        if length(x)>d2: # if x is a list, and not a single point, we unpack recursively
            return np.array([u(x_i,t,alpha,time_delta) for x_i in x])
        assert (length(x)) == (d2 if d2>1 else 0)
        x = [x] if length(x) == 0 else x
        return u_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha)
    return (f,u)

# varying conductivity k=k(u)
t = sp.symbols('t')
x = sp.symbols('x')
alpha = sp.symbols('alpha')
var_k = []
k_0 = 1
# extravark: highly varying conductivity
xvar_k = []

# Testing:
u = t*x*x
f,u = manufacture_solution(u,t,[x], k=alpha*u, alpha_var=alpha, d1=1,d2=1)
var_k.append((f,u))

u = sp.sin(alpha*t+x)
f,u = manufacture_solution(u,t,[x], k=u*u, alpha_var=alpha, d1=1,d2=1)
var_k.append((f,u))

u = t*x*x
f,u = manufacture_solution(u,t,[x], k=alpha*u/10+k_0, alpha_var=alpha, d1=1,d2=1)
var_k.append((f,u))

# actual ones:
u = sp.sin(5*alpha*t+x)+sp.sin(alpha*x)/2
f,u = manufacture_solution(u,t,[x], k=sp.sin(2*alpha*u)/2 + 1, alpha_var=alpha, d1=1,d2=1)
var_k.append((f,u))
u = sp.sin(5*alpha*t+x)+sp.sin(alpha*x)/2
fx,ux = manufacture_solution(u,t,[x], k=sp.sin(2*alpha*u) + 1, alpha_var=alpha, d1=1,d2=1)
xvar_k.append((fx,ux))

u = sp.cos(x*alpha)*(sp.exp(-t)+sp.exp((t-1))*2)
f,u = manufacture_solution(u,t,[x], k=alpha*u/10+k_0, alpha_var=alpha, d1=1,d2=1)
var_k.append((f,u))
u = sp.cos(x*alpha)*(sp.exp(-t)+sp.exp((t-1))*2)
fx,ux = manufacture_solution(u,t,[x], k=alpha*u/2+k_0, alpha_var=alpha, d1=1,d2=1)
xvar_k.append((fx,ux))

u = 1-2*x + alpha*x**2 -t*2 + x*t*2
f,u = manufacture_solution(u,t,[x], k=sp.exp(u/10), alpha_var=alpha, d1=1,d2=1)
var_k.append((f,u))
u = 1-2*x + alpha*x**2 -t*2 + x*t*2
fx,ux = manufacture_solution(u,t,[x], k=sp.exp(u/4), alpha_var=alpha, d1=1,d2=1)
xvar_k.append((fx,ux))

u = 1/(1+x) + (alpha+x)/(5*t+1)**0.5
f,u = manufacture_solution(u,t,[x], k=sp.exp(-u/10), alpha_var=alpha, d1=1,d2=1)
var_k.append((f,u))
u = 1/(1+x) + (alpha+x)/(5*t+1)**0.5
fx,ux = manufacture_solution(u,t,[x], k=sp.exp(-u/4), alpha_var=alpha, d1=1,d2=1)
xvar_k.append((fx,ux))



# BP heat
y = sp.symbols('y')
z = sp.symbols('z')
dimred = []

u1 = sp.sin(x+alpha*y+2*z + 4*alpha*t)
f,u = manufacture_solution(u1,t,[x,y,z], alpha_var=alpha, d1=3,d2=2)
dimred.append((f,u))

u2 = (sp.cos(alpha*x)+sp.sin(alpha+z)+sp.sin(t+y))*(sp.exp(-t)+sp.exp((t-1))*2)
f,u = manufacture_solution(u2,t,[x,y,z], alpha_var=alpha, d1=3,d2=2)
dimred.append((f,u))

u3 = 1-2*x+3*y-z + alpha*x**2 + alpha*x*z*y -t*2 + x*t*2 - 4*z*t*y
f,u = manufacture_solution(u3,t,[x,y,z], alpha_var=alpha, d1=3,d2=2)
dimred.append((f,u))

u4 = 1/(1+x+y+z) + (alpha+x-y-z)/(5*t+1)**0.5
f,u = manufacture_solution(u4,t,[x,y,z], alpha_var=alpha, d1=3,d2=2)
dimred.append((f,u))

u5 = 5*t*alpha*(x+y-z)**2+2**(x-y+z-alpha)*5*t**0.5
f,u = manufacture_solution(u5,t,[x,y,z], alpha_var=alpha, d1=3,d2=2)
dimred.append((f,u))

u6 = 1/(1+x+y+z) + sp.log(alpha+x+y+z+2)/(5*t+1)**0.5
f,u = manufacture_solution(u6,t,[x,y,z], alpha_var=alpha, d1=3,d2=2)
dimred.append((f,u))

#u3 = 1-2*x+3*y**2-z + alpha*x**2 + alpha*x**1.3*z*y**2 -t*2 + x*t*2 - 4*z*t*y
u1 = u1.subs(z,0)
u2 = u2.subs(z,0)
u3 = u3.subs(z,1)
u4 = u4.subs(z,0)
u5 = u5.subs(z,0)
u6 = u6.subs(z,0)
lowdimred = []
f_1,u_1 = manufacture_solution(u1,t,[x,y], alpha_var=alpha, d1=2,d2=1)
f_2,u_2 = manufacture_solution(u1,t,[x,y], alpha_var=alpha, d1=2,d2=2)
lowdimred.append((f_1,u_1,f_2,u_2))
f_1,u_1 = manufacture_solution(u2,t,[x,y], alpha_var=alpha, d1=2,d2=1)
f_2,u_2 = manufacture_solution(u2,t,[x,y], alpha_var=alpha, d1=2,d2=2)
lowdimred.append((f_1,u_1,f_2,u_2))
f_1,u_1 = manufacture_solution(u3,t,[x,y], alpha_var=alpha, d1=2,d2=1)
f_2,u_2 = manufacture_solution(u3,t,[x,y], alpha_var=alpha, d1=2,d2=2)
lowdimred.append((f_1,u_1,f_2,u_2))
f_1,u_1 = manufacture_solution(u4,t,[x,y], alpha_var=alpha, d1=2,d2=1)
f_2,u_2 = manufacture_solution(u4,t,[x,y], alpha_var=alpha, d1=2,d2=2)
lowdimred.append((f_1,u_1,f_2,u_2))
f_1,u_1 = manufacture_solution(u5,t,[x,y], alpha_var=alpha, d1=2,d2=1)
f_2,u_2 = manufacture_solution(u5,t,[x,y], alpha_var=alpha, d1=2,d2=2)
lowdimred.append((f_1,u_1,f_2,u_2))
f_1,u_1 = manufacture_solution(u6,t,[x,y], alpha_var=alpha, d1=2,d2=1)
f_2,u_2 = manufacture_solution(u6,t,[x,y], alpha_var=alpha, d1=2,d2=2)
lowdimred.append((f_1,u_1,f_2,u_2))



# Elasticity solutions:
t = sp.symbols('t')
x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')
alpha = sp.symbols('alpha')
ELsols = [
            {'u':
                    [sp.sin(np.pi*(x+alpha*y))*sp.cos(alpha*t),
                     sp.cos(np.pi*(x+alpha*y))*sp.sin(alpha*t)],
             't_var':t,
             'x_vars':[x,y],
             'alpha_var':alpha,
            },
            {'u':
                    [sp.exp((-t*x**2 + y**2)/(1+alpha+t**2)),
                    sp.exp((+t*x**2 - y**2)/(1+alpha+t**2))],
             't_var':t,
             'x_vars':[x,y],
             'alpha_var':alpha,
            },
            {'u':
                    [(x**3) + (y**2)*((t+0.5)**1.5) + x*y*alpha,
                     (x**2) + (y**3)*((t+0.5)**1.1) + x*y*alpha],
             't_var':t,
             'x_vars':[x,y],
             'alpha_var':alpha,
            }
         ]
ELsols3d = [
            {'u':
                    [sp.sin(np.pi*(x+alpha*y+(1+alpha)/2*z))*sp.cos(alpha*t),
                     sp.cos(np.pi*(x+alpha*y+(1+alpha)/2*z))*sp.sin(alpha*t),
                    -sp.cos(np.pi*(x+alpha*y+(1+alpha)/2*z))*sp.sin(alpha*t)],
             't_var':t,
             'x_vars':[x,y,z],
             'alpha_var':alpha,
            },
            {'u':
                    [sp.exp((-t*x**2 + y**2 + z**2)/(1+alpha+t**2)),
                     sp.exp((+t*x**2 - y**2 + z**2)/(1+alpha+t**2)),
                     sp.exp((+t*x**2 + y**2 - z**2)/(1+alpha+t**2))],
             't_var':t,
             'x_vars':[x,y,z],
             'alpha_var':alpha,
            },
            {'u':
                    [(x**3) + (y**2)*((t+0.5)**1.5) + x*y*alpha + (t+0.5)**0.5*z**2 + z*(x+y)*alpha,
                     (x**2) + (y**3)*((t+0.5)**1.1) - x*y*alpha + (t+0.5)**0.5*z**2 + z*(x-y)*alpha,
                     (x**2) + (y**2)*((t+0.5)**1.1) + x*y*alpha + (t+0.5)**0.5*z**3 + z*(-x+y)*alpha
                     ],
             't_var':t,
             'x_vars':[x,y,z],
             'alpha_var':alpha,
            }
         ]

def manufacture_elasticity_solution(u, x_vars, t_var, alpha_var=None, d1=2, d2=2, nu=0.25, static=False, non_linear=False):
    '''Manufactures f from the elasticity equation given u (list of sympy equations)
    returns functions with unused dimensions=0'''
    print('manufacturing solution, from', d1, 'to', d2, 'dimensions.')
    assert d1>=d2
    assert d1<=3
    assert d2>=2
    assert len(u) == d1
    if d1 == 2:
        epsilon_bar = np.array([
            u[0].diff(x_vars[0]),
            u[1].diff(x_vars[1]),
            u[1].diff(x_vars[0])+u[0].diff(x_vars[1])])
        C = np.array([
            [1,nu,0],
            [nu,1,0],
            [0,0,(1-nu)/2]
            ])/(1-nu**2)
        E = 1
        if non_linear:
            e = (epsilon_bar[0]**2+epsilon_bar[1]**2+u[1].diff(x_vars[0])**2+u[0].diff(x_vars[1])**2)**0.5 # norm of epsilon
            A, c = 10,20
            E = A / (2*(c+e)**0.5)
        sigma_bar = E*C @ epsilon_bar

        # static: f = -Div(sigma) (remember sigma != sigma_bar)
        f = np.array([-sigma_bar[0].diff(x_vars[0]) -sigma_bar[2].diff(x_vars[1]),
                      -sigma_bar[2].diff(x_vars[0]) -sigma_bar[1].diff(x_vars[1])])
    if d1 == 3:
        epsilon_bar = np.array([
            u[0].diff(x_vars[0]),
            u[1].diff(x_vars[1]),
            u[2].diff(x_vars[2]),
            u[1].diff(x_vars[2])+u[2].diff(x_vars[1]),
            u[2].diff(x_vars[0])+u[0].diff(x_vars[2]),
            u[0].diff(x_vars[1])+u[1].diff(x_vars[0])
            ])
        C = np.array([
            [1-nu,nu,nu,0,0,0],
            [nu,1-nu,nu,0,0,0],
            [nu,nu,1-nu,0,0,0],
            [0,0,0,1/2-nu,0,0],
            [0,0,0,0,1/2-nu,0],
            [0,0,0,0,0,1/2-nu]
            ]) / ((1+nu)*(1-2*nu))
        sigma_bar = C @ epsilon_bar
        assert not non_linear

        # static: f = -Div(sigma) (remember sigma != sigma_bar)
        f = -np.array([
                sigma_bar[0].diff(x_vars[0]) + sigma_bar[5].diff(x_vars[1]) + sigma_bar[4].diff(x_vars[2]),
                sigma_bar[5].diff(x_vars[0]) + sigma_bar[1].diff(x_vars[1]) + sigma_bar[3].diff(x_vars[2]),
                sigma_bar[4].diff(x_vars[0]) + sigma_bar[3].diff(x_vars[1]) + sigma_bar[2].diff(x_vars[2]),
                ])
             

    # add transient term; now f = u_tt - Div(sigma)
    if not static:
        f += np.array([ui.diff(t_var,t_var) for ui in u])

    # make lambda functions
    f_temp = lambdify([*[x_vars],t_var,alpha_var],f[:d2], "numpy")
    u_temp = lambdify([*[x_vars],t_var,alpha_var],u[:d2], "numpy")
    w_temp = lambdify([*[x_vars],t_var,alpha_var],[ui.diff(t_var) for ui in u[:d2]], "numpy")

    # For controlling and/or plotting level of nonlinearity
    if non_linear:
        e_temp = lambdify([*[x_vars],t_var,alpha_var],e, "numpy")
        E_temp = lambdify([*[x_vars],t_var,alpha_var],E, "numpy")
        global E_max, E_min, e_max, e_min
        E_max = 0
        E_min = 100
        e_max = 0
        e_min = 100
    
    # make functions
    def f(x,t,alpha,time_delta=0):
        if non_linear:
            E_val = E_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha) 
            e_val = e_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha) 
            global E_min, E_max, e_min, e_max
            if E_val < E_min:
                E_min = E_val
            if E_val > E_max:
                E_max = E_val
            if e_val < e_min:
                e_min = e_val
            if e_val > e_max:
                e_max = e_val
        x = [x] if length(x) == 0 else x
        return f_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha)
    def u(x,t,alpha,time_delta=0):
        if length(x)>d2: # if x is a list, and not a single point, we unpack recursively
            return np.array([u(x_i,t,alpha,time_delta) for x_i in x])
        assert (length(x)) == (d2 if d2>1 else 0)
        x = [x] if length(x) == 0 else x
        return u_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha)
    def w(x,t,alpha,time_delta=0):
        if length(x)>d2: # if x is a list, and not a single point, we unpack recursively
            return np.array([w(x_i,t,alpha,time_delta) for x_i in x])
        assert (length(x)) == (d2 if d2>1 else 0)
        x = [x] if length(x) == 0 else x
        return w_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha)
    return (f,u, w)

def get_elastic_nonlinearity():
    global E_min, E_max, e_min, e_max
    #print('Minimum E:', E_min)
    #print('Maximum E:', E_max)
    return E_min, E_max, e_min, e_max

def get_heat_nonlinearity():
    global k_min, k_max, u_min, u_max
    #print('Minimum E:', E_min)
    #print('Maximum E:', E_max)
    res = k_min, k_max, u_min, u_max
    k_min = 100
    u_min = 100
    u_max = -100
    k_max = -100
    return res
