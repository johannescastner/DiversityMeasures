#!/usr/bin/env python

# implement the example graphs/integral from pyx
from pylab import *
from matplotlib.patches import Polygon

def func1(x):
    return 5*x*x

def func2(x):
    return 20*x*x

def func3(x):
    return 30 + x*x

ax = subplot(111)

a, b = -4, 4 # integral area
x = arange(-3, 3, 0.01)

y = np.row_stack((func1(x), func2(x), func3(x)))

y1, y2, y3 = func1(x), func2(x), func3(x)

plot(x, y, linewidth=1)

# make the shaded region
ix = arange(a, b, 0.01)
iy = func(ix)
verts = [(a,0)] + list(zip(ix,iy)) + [(b,0)]
poly = Polygon(verts, facecolor='0.8', edgecolor='k')
ax.add_patch(poly)

text(0.5 * (a + b), 30,
     r"$\int_a^b f(x)\mathrm{d}x$", horizontalalignment='center',
     fontsize=20)

axis([-3,3, 0, 180])
figtext(0.9, 0.05, 'x')
figtext(0.1, 0.9, 'y')
ax.set_xticks((a,b))
ax.set_xticklabels(('a','b'))
ax.set_yticks([])
show()
