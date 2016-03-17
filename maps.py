import numpy as np
from matplotlib.pyplot import *
from matplotlib.font_manager import FontProperties
from itertools import cycle
def arrowplot(axes, x, y, narrs=30, dspace=0.1, direc='pos', \
                          hl=0.01, hw=4, color='red', label=""): 
    ''' narrs  :  Number of arrows that will be drawn along the curve

        dspace :  Shift the position of the arrows along the curve.
                  Should be between 0. and 1.

        direc  :  can be 'pos' or 'neg' to select direction of the arrows

        hl     :  length of the arrow head 

        hw     :  width of the arrow head        

        c      :  color of the edge and face of the arrow head  
    '''

    # r is the distance spanned between pairs of points
    r = [0]
    for i in range(1,len(x)):
        dx = x[i]-x[i-1] 
        dy = y[i]-y[i-1] 
        r.append(np.sqrt(dx*dx+dy*dy))
    r = np.array(r)

    # rtot is a cumulative sum of r, it's used to save time
    rtot = []
    for i in range(len(r)):
        rtot.append(r[0:i].sum())
    rtot.append(r.sum())

    # based on narrs set the arrow spacing
    aspace = r.sum() / narrs

    if direc is 'neg':
        dspace = -1.*abs(dspace) 
    else:
        dspace = abs(dspace)

    arrowData = [] # will hold tuples of x,y,theta for each arrow
    arrowPos = aspace*(dspace) # current point on walk along data
                                 # could set arrowPos to 0 if you want
                                 # an arrow at the beginning of the curve
    ndrawn = 0
    rcount = 1 
    while arrowPos < r.sum() and ndrawn < narrs:
        x1,x2 = x[rcount-1],x[rcount]
        y1,y2 = y[rcount-1],y[rcount]
        da = arrowPos-rtot[rcount]
        theta = np.arctan2((x2-x1),(y2-y1))
        ax = np.sin(theta)*da+x1
        ay = np.cos(theta)*da+y1
        arrowData.append((ax,ay,theta))
        ndrawn += 1
        arrowPos+=aspace
        while arrowPos > rtot[rcount+1]: 
            rcount+=1
            if arrowPos > rtot[-1]:
                break

    # could be done in above block if you want
    for ax,ay,theta in arrowData:
        # use aspace as a guide for size and length of things
        # scaling factors were chosen by experimenting a bit

        dx0 = np.sin(theta)*hl/2. + ax
        dy0 = np.cos(theta)*hl/2. + ay
        dx1 = -1.*np.sin(theta)*hl/2. + ax
        dy1 = -1.*np.cos(theta)*hl/2. + ay

        if direc is 'neg' :
          ax0 = dx0 
          ay0 = dy0
          ax1 = dx1
          ay1 = dy1 
        else:
          ax0 = dx1 
          ay0 = dy1
          ax1 = dx0
          ay1 = dy0 

        axes.annotate('', xy=(ax0, ay0), xycoords='data',
                xytext=(ax1, ay1), textcoords='data',
                      arrowprops=dict( headwidth=hw, frac=1., ec=color, fc=color))


    axes.plot(x,y, color = color, label=label)
    #axes.set_xlim(x.min()*.9,x.max()*1.1)
    #axes.set_ylim(y.min()*.9,y.max()*1.1)

class BaryCentricMap(object):
    def __init__(self):
        self.x1 = np.array([1.0, 0.0])
        self.x2 = np.array([0.0, 0.0])
        self.x3 = np.array([0.5, np.sqrt(3.0/4.0)])
        self.x4 = np.array([1.0/3.0, 0.0])
        self.color = cycle(['r', 'g', 'b', 'Chocolate', 'Brown', 'Olive', 'LightSeaGreen', 'YellowGreen'])
        self.B = np.array([[-2.0, 0.5],[0.0, 2.598076211353316]])

    def plot_triangle(self):
        font_ = FontProperties()
        font_.set_family('sans-serif')
        font_.set_weight('normal')
        font_.set_style('italic')
        alpha = 0.8
        self.fig = figure()
        
        alphal = 0.5
        plot((self.x1[0], self.x2[0]), (self.x1[1], self.x2[1]), color='k', alpha=alphal)
        plot((self.x2[0], self.x3[0]), (self.x2[1], self.x3[1]), color='k', alpha=alphal)
        plot((self.x3[0], self.x1[0]), (self.x3[1], self.x1[1]), color='k', alpha=alphal)
        plot((self.x3[0], self.x4[0]), (self.x3[1], self.x4[1]), color='k', alpha=alphal)
        self.ax = gca()
        xlim(-0.2, 1.2)
        ylim(-0.2, 1.1)
        gca().annotate(r'One component', xy=self.x1, xytext=(0.85, -0.05), 
                       fontproperties=font_, alpha=alpha)
        gca().annotate(r'Two component', xy=self.x2, xytext=(-0.15, -0.05), 
                       fontproperties=font_, alpha=alpha)
        gca().annotate(r'Three component', xy=self.x3, xytext=(0.35, 0.90), 
                       fontproperties=font_, alpha=alpha)
        m = (self.x3[1] - self.x4[1])/(self.x3[0] - self.x4[0])
        y = 0.1
        x = self.x4[0] + (1.0/m)*y
        gca().annotate(r'Plane strain', xy=np.array([x, y]), xytext=(0.5, -0.1), 
                       fontproperties=font_,arrowprops=dict(facecolor='black',lw=0.5, arrowstyle="->",), alpha=alpha)

        m = (self.x3[1] - self.x1[1])/(self.x3[0] - self.x1[0])
        y = 0.6
        x = self.x1[0] + (1.0/m)*y
        dx = 0.02
        gca().annotate(r'Axisymmetric Expansion', xy=np.array([x, y]), xytext=np.array([x+dx, y-1.0/m*dx]), fontproperties=font_, rotation=-55, alpha=alpha)

        m = self.x3[1]/self.x3[0]
        y = 0.6
        x = (1.0/m)*y - 0.27
        dx = 0.02
        gca().annotate(r'Axisymmetric Contraction', xy=np.array([x, y]), xytext=np.array([x-dx, y+1.0/m*dx]), fontproperties=font_, rotation=55, alpha=alpha)

        grid(False)
        gca().axis('off')
        gcf().patch.set_visible(False)
        tight_layout()
        
    def calc_trajectory(self, aij):
        n = np.shape(aij)[2]
        x = np.zeros(n)
        y = np.zeros(n)
        for i in range(n):
            aijn = aij[:, :, i]
            w, v = np.linalg.eig(aijn)
            w.sort()
            w_sorted = w[::-1]
            X = self.B.dot(w_sorted[1:].T) + self.x3
            x[i] = X[0]
            y[i] = X[1]
        return x[1:], y[1:]

    def plot_trajectory(self, x, y, label=""):
        arrowplot(self.ax, x, y, color=next(self.color), label=label)
        legend(loc='best', framealpha=0.5)
    def save(self, name="barycentric"):
        savefig("%s.pdf"%name)
        savefig("%s.png"%name)
    

class AnisotropyInvariantMap(BaryCentricMap):
    def plot_triangle(self):
        
        font_ = FontProperties()
        font_.set_family('sans-serif')
        font_.set_weight('normal')
        font_.set_style('italic')
        self.fig = figure()
        alpha = 0.8
        alphal = 0.5
        third_range = np.linspace(-0.0277, 0.21, 10000)
        second_upper = 2*third_range + 2.0/9.0
        plot(third_range, second_upper, color='k', alpha=alphal)
        
        second_right = 1.5*(abs(third_range)*4.0/3.0)**(2.0/3.0)
        plot(third_range, second_right, color='k', alpha=alphal)
        connectionstyle="arc3,rad=.1"
        lw = 0.5
        plot(np.array([0.0, 0.0]), np.array([0.0, 2.0/9.0]), color='k', alpha=alphal)
        gca().annotate(r'Isotropic limit', xy=np.array([0, 0]), xytext=np.array([0.05, 0.02]), fontproperties=font_, rotation=0, alpha=alpha, arrowprops=dict(arrowstyle="->", connectionstyle=connectionstyle, lw=lw))
        gca().annotate(r'Plane strain', xy=np.array([0, 1.0/9.0/2]), xytext=np.array([0.07, 0.07]), fontproperties=font_, rotation=0, alpha=alpha, arrowprops=dict(arrowstyle="->", connectionstyle=connectionstyle, lw=lw))
        gca().annotate(r'One component limit', xy=np.array([third_range[-1], second_upper[-1]]), xytext=np.array([0.00, 0.6]), fontproperties=font_, rotation=0, alpha=alpha, arrowprops=dict(arrowstyle="->", connectionstyle=connectionstyle, lw=lw))
        gca().annotate(r'Axisymmetric contraction', xy=np.array([third_range[1000/2], second_right[1000/2]]), xytext=np.array([0.09, 0.12]), fontproperties=font_, rotation=0, alpha=alpha, arrowprops=dict(arrowstyle="->", connectionstyle=connectionstyle, lw=lw))
        gca().annotate(r'Two component axisymmetric', xy=np.array([third_range[0], second_right[0]]), xytext=np.array([0.11, 0.17]), fontproperties=font_, rotation=0, alpha=alpha, arrowprops=dict(arrowstyle="->", connectionstyle=connectionstyle, lw=lw))
        gca().annotate(r'Two component plane strain', xy=np.array([0, 2.0/9.0]), xytext=np.array([0.13, 0.22]), fontproperties=font_, rotation=0, alpha=alpha, arrowprops=dict(arrowstyle="->", connectionstyle=connectionstyle, lw=lw))
        gca().annotate(r'Axisymmetric expansion', xy=np.array([third_range[4000], second_right[4000]]), xytext=np.array([0.15, 0.27]), fontproperties=font_, rotation=0, alpha=alpha, arrowprops=dict(arrowstyle="->", connectionstyle=connectionstyle, lw=lw))
        gca().annotate(r'Two component', xy=np.array([third_range[6000], second_upper[6000]]), xytext=np.array([0.05, 0.5]), fontproperties=font_, rotation=0, alpha=alpha, arrowprops=dict(arrowstyle="->", connectionstyle=connectionstyle, lw=lw))


        self.ax = gca()
        xlim(-0.05, 0.3)
        ylim(0.0, 0.7)
        grid(False)
        gca().axis('off')
        gcf().patch.set_visible(False)
        tight_layout()
        
    def calc_trajectory(self, aij):
        n = np.shape(aij)[2]
        second = np.zeros(n)
        third = np.zeros(n)
        for i in range(n):
            aijn = aij[:, :, i]
            w, v = np.linalg.eig(aijn)
            w.sort()
            w_sorted = w[::-1]
            X = self.B.dot(w_sorted[1:].T) + self.x3
            second[i] = 2.0*(w[0]**2 + w[0]*w[1] + w[1]**2)
            third[i] = -3.0*w[0]*w[1]*(w[0] + w[1])
        return third[1:], second[1:]

    def calc_from_barycentric(self, x, y):
        n = np.size(x)
        second = np.zeros(n)
        third = np.zeros(n)
        
        for i in range(n):
            A = np.array([[self.x1[0] - self.x3[0], self.x2[0] - self.x3[0]],[self.x1[1] - self.x3[1], self.x2[1] - self.x3[1]]])
            rhs = np.array([x[i] - self.x3[0], y[i] - self.x3[1]])
            C = np.linalg.solve(A, rhs)
            second[i] = 2.0/3.0*C[0]**2 + 1.0/3.0*C[0]*C[1] + 1.0/6.0*C[1]**2
            third[i] = 2.0/9.0*C[0]**3 + 1.0/6.0*C[0]**2*C[1] - 1.0/12.0*C[0]*C[1]**2 - 1.0/36.0*C[1]**3
        return third, second

if __name__ == "__main__":
    pass
