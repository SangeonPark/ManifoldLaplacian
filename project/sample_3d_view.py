import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def sample_sphere(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def sample_torus(npoints):
    # sample points from S^1 \times S^1
    theta = 2* np.pi*np.random.random_sample(npoints)
    phi = 2*np.pi*np.random.random_sample(npoints)
    c, a = 2, 1

    vec = np.zeros(3, npoints)
    vec[0, :] = (c + a*np.cos(theta)) * np.cos(phi)
    vec[1, :] = (c + a*np.cos(theta)) * np.sin(phi)
    vec[2, :] = a * np.sin(theta)

    return vec

def visualize_grid_sphere():
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
    return fig, ax



def visualize_grid_torus(n):
    theta = np.linspace(0, 2.*np.pi, n)
    phi = np.linspace(0, 2.*np.pi, n)
    theta, phi = np.meshgrid(theta, phi)
    c, a = 2, 1
    x = (c + a*np.cos(theta)) * np.cos(phi)
    y = (c + a*np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
    #ax1 = fig.add_subplot(121, projection='3d')
    #ax1.set_zlim(-3,3)
    #ax1.plot_surface(x, y, z, rstride=5, cstride=5, color='k', edgecolors='w')
    #ax1.view_init(36, 26)
    #ax2 = fig.add_subplot(122, projection='3d')
    #ax2.set_zlim(-3,3)
    #ax2.plot_surface(x, y, z, rstride=5, cstride=5, color='k', edgecolors='w')
    #ax2.view_init(0, 0)
    #ax2.set_xticks([])
    return







phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

xi, yi, zi = sample_spherical(100)

fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)
