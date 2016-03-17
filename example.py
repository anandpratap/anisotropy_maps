import scipy.io as io
from maps import BaryCentricMap, AnisotropyInvariantMap
import matplotlib.pyplot as plt

data = io.loadmat("test.mat")

Xstar = data['Xstar']
xstar = Xstar[0,:]
ystar = Xstar[1,:]

aijdns = data['aijdns']
aijrans = data['aijrans']

bary = BaryCentricMap()
bary.plot_triangle()
xdns, ydns = bary.calc_trajectory(aijdns)
xrans, yrans = bary.calc_trajectory(aijrans)
bary.plot_trajectory(xdns, ydns, label='DNS')
bary.plot_trajectory(xrans, yrans, label='Base')
bary.plot_trajectory(xstar, ystar, label='MAP')
bary.save('bary_example')

anisotropy_map = AnisotropyInvariantMap()
anisotropy_map.plot_triangle()
xstar, ystar = anisotropy_map.calc_from_barycentric(xstar, ystar)
xdns, ydns = anisotropy_map.calc_trajectory(aijdns)
xrans, yrans = anisotropy_map.calc_trajectory(aijrans)

anisotropy_map.plot_trajectory(xdns, ydns, label='DNS')
anisotropy_map.plot_trajectory(xrans, yrans, label='Base')
anisotropy_map.plot_trajectory(xstar, ystar, label='MAP')
anisotropy_map.save('lumley_example')

plt.show()
