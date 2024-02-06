import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def genus_two(**kwargs):
    def solve_for_z(x, y):
        x, y = map(lambda x: np.array(x).astype('complex'), [x, y])
        return np.sqrt(
            0.01 - ((x**2 + y**2) ** 2 - .75*x**2 + .75*y**2)**2
        )
    aa = np.linspace(-1, 1, 200)
    xx, yy = np.meshgrid(aa, aa)
    xx = xx.flatten()
    yy = yy.flatten()
    zz = solve_for_z(xx, yy)
    ix_on_surface = np.isreal(zz)

    data_pos = np.vstack([
        xx[ix_on_surface],
        yy[ix_on_surface],
        np.real(zz[ix_on_surface]),
    ]).T

    data_neg = np.vstack([
        xx[ix_on_surface],
        yy[ix_on_surface],
        -np.real(zz[ix_on_surface]),
    ]).T
    data = np.vstack([data_pos, data_neg])
    #chosen_ix = choose_n_farthest_points(data, 1500, 42)
    data_orig = data[:]
    return data_orig


def genus_three_temp(sr, lr, c):
    A = sr**2 + lr**2
    B = lr**2 - sr**2
    def solve_for_sqrt(x, y):
        x, y = map(lambda x: np.array(x).astype('complex'), [x, y])
        k = x**2 + y**2
        p = k
        q = B
        r = -2*A*k + B**2 - c
        return np.sqrt(
            2*p*q + q**2 - r
        )
    
    def solve_for_z(x, y, sqrt):
        x, y, sqrt = map(lambda x: np.array(x).astype('complex'), [x, y, sqrt])
        k = x**2 + y**2
        p = k
        q = B
        return np.sqrt(
           sqrt-p-q
        )
        
    aa = np.linspace(-20, 20, 500)
    xx, yy = np.meshgrid(aa, aa)
    xx = xx.flatten()
    yy = yy.flatten()
    sqrt = solve_for_sqrt(xx, yy)
    ix_sqrt_real = np.isreal(sqrt)
    #xx = xx[ix_sqrt_real]
    #yy = yy[ix_sqrt_real]
    #sqrt = sqrt[ix_sqrt_real]
    
    zz_first = solve_for_z(xx, yy, sqrt)
    zz_second = solve_for_z(xx, yy, -sqrt)
    
    first_ix_on_surface = np.isreal(zz_first)
    second_ix_on_surface = np.isreal(zz_second)
    
    

    
    data_first_pos = np.vstack([
        xx[first_ix_on_surface],
        yy[first_ix_on_surface],
        np.real(zz_first[first_ix_on_surface]),
    ]).T

    data_first_neg = np.vstack([
        xx[first_ix_on_surface],
        yy[first_ix_on_surface],
        -np.real(zz_first[first_ix_on_surface]),
    ]).T
    
    data_second_pos = np.vstack([
        xx[second_ix_on_surface],
        yy[second_ix_on_surface],
        np.real(zz_second[second_ix_on_surface]),
    ]).T

    data_second_neg = np.vstack([
        xx[second_ix_on_surface],
        yy[second_ix_on_surface],
        -np.real(zz_second[second_ix_on_surface]),
    ]).T
    
    
    data = np.vstack([data_first_pos, data_first_neg,data_second_pos, data_second_neg])
    
    
    
    #chosen_ix = choose_n_farthest_points(data, 1500, 42)
    data_orig = data[:]
    return data_orig

def genus_three(**kwargs):
    def solve_for_z(x, y):
        x, y = map(lambda x: np.array(x).astype('complex'), [x, y])
        return np.sqrt(
            0.1 - ((x+.1)*(x-.75)**2*(x-2.25)**2*(x-3.1) + .75*y**2)**2
        )
    #xcoords = np.concatenate((np.linspace(-4.5,-2.9,1000), np.linspace(-2.8999,4,100)))
    xcoords = np.linspace(-4,4,430)
    ycoords = np.linspace(-1.2, 1.2, 50)
    xx, yy = np.meshgrid(xcoords, ycoords)
    xx = xx.flatten()
    yy = yy.flatten()
    zz = solve_for_z(xx, yy)
    ix_on_surface = np.isreal(zz)

    data_pos = np.vstack([
        xx[ix_on_surface],
        yy[ix_on_surface],
        np.real(zz[ix_on_surface]),
    ]).T

    data_neg = np.vstack([
        xx[ix_on_surface],
        yy[ix_on_surface],
        -np.real(zz[ix_on_surface]),
    ]).T
    data = np.vstack([data_pos, data_neg])
    #chosen_ix = choose_n_farthest_points(data, 1500, 42)
    data_orig = data[:]
    return data_orig

def sample_sphere(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T


def sample_ellipsoid(npoints, a=1,b=2,c=1.5):
    theta = 2* np.pi*np.random.random_sample(npoints)
    phi = 2*np.pi*np.random.random_sample(npoints)
    
    vec = np.zeros((3, npoints))
    vec[0, :] = ( a*np.sin(theta)) * np.cos(phi)
    vec[1, :] = ( b*np.sin(theta)) * np.sin(phi)
    vec[2, :] = c * np.cos(theta)
    return vec.T
    

def sample_torus(npoints, outer, inner):
    # sample points from S^1 \times S^1
    theta = 2* np.pi*np.random.random_sample(npoints)
    phi = 2*np.pi*np.random.random_sample(npoints)
    c, a = outer, inner

    vec = np.zeros((3, npoints))
    vec[0, :] = (c + a*np.cos(theta)) * np.cos(phi)
    vec[1, :] = (c + a*np.cos(theta)) * np.sin(phi)
    vec[2, :] = a * np.sin(theta)

    return vec.T

def visualize_grid_sphere(scale=1):
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))*scale
    y = np.outer(np.sin(theta), np.sin(phi))*scale
    z = np.outer(np.cos(theta), np.ones_like(phi))*scale
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'auto'})
    ax.set_box_aspect([1,1,1])
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1, linewidth=1, alpha=0.3)
    return fig, ax



def visualize_grid_torus(n, outer, inner):
    theta = np.linspace(0, 2.*np.pi, n)
    phi = np.linspace(0, 2.*np.pi, n)
    theta, phi = np.meshgrid(theta, phi)
    c, a = outer, inner
    x = (c + a*np.cos(theta)) * np.cos(phi)
    y = (c + a*np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1,linewidth=1, alpha=0.3)
    minbound = -outer
    maxbound = outer
    ax.auto_scale_xyz([minbound, maxbound], [minbound, maxbound], [minbound, maxbound])
    #ax.set_box_aspect([1,1,1])
    #ax1 = fig.add_subplot(121, projection='3d')
    #ax1.set_zlim(-3,3)
    #ax1.plot_surface(x, y, z, rstride=5, cstride=5, color='k', edgecolors='w')
    #ax1.view_init(36, 26)
    #ax2 = fig.add_subplot(122, projection='3d')
    #ax2.set_zlim(-3,3)
    #ax2.plot_surface(x, y, z, rstride=5, cstride=5, color='k', edgecolors='w')
    #ax2.view_init(0, 0)
    #ax2.set_xticks([])
    return fig, ax








