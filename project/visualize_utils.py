import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from mpl_toolkits.mplot3d import axes3d

mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['figure.dpi'] = 300
plt.rcParams["figure.figsize"] = (6,6)
plt.rcParams.update({'font.size': 24})

import numpy as np


class Visualizer(object):
    """docstring for Visualizer"""
    def __init__(self, data, tangents, eigvectors, bounds, view_params):
        super(Visualizer, self).__init__()
        self.tangents = tangents
        self.eigvectors = eigvectors
        self.bounds = bounds
        self.view_params = view_params
        self.data = data
        self.npoints = data.shape[0]
    
    def draw_canvas(self):
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
        #ax.set_box_aspect([2,2,1])
        #fig, ax = visualize_grid_torus(30,2,1)
        #ax.scatter(data[:,0], data[:, 1], data[:,2], s=1, c=sc, cmap='viridis',zorder=10, alpha=0.5)
        ax.view_init(self.view_params[0], self.view_params[1])


        xminbound, xmaxbound, yminbound, ymaxbound, zminbound, zmaxbound = self.bounds
        ax.auto_scale_xyz([xminbound, xmaxbound], [yminbound, ymaxbound], [zminbound, zmaxbound])


        self.fig = fig
        self.ax = ax





    def draw_points(self, size, alpha):
        self.ax.scatter(self.data[:,0], self.data[:, 1],self.data[:,2], s=size, zorder=10, alpha=alpha)






    def draw_vectors(self, index, nvecs, vecfield_scale, color="C0", linewidth=0.6):
        assert index < self.eigvectors.shape[1]
        eigvector = self.eigvectors[:, index]
        eigvector = eigvector.reshape(-1,2)
        vecfield = np.zeros((self.npoints, 3))
        for i in range(self.npoints):
            vecfield[i] = self.tangents[i, 0]* eigvector[i, 0] +  self.tangents[i, 1]* eigvector[i, 1] 
        visualizer = np.zeros((self.npoints, 6))
        visualizer[:,:3] = self.data
        visualizer[:,3:] = vecfield*vecfield_scale
        X, Y, Z, U, V, W = zip(*visualizer[np.random.choice(np.arange(0,self.npoints), nvecs), :])
        self.ax.quiver(X, Y, Z, U, V, W, color=color, linewidth=linewidth)



