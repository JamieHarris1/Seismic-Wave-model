import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class AnimFct():
    """ Animate a 2 variable function from a list of functions
        Data : times : list of time values 
               data : list of array of function values of shape: NX, NY, Nf 
    """
    def __init__(self,Xmin=0, Xmax=1, Ymin=0, Ymax=1):
        """ Initialiser:
            : param Xmin, Xmax : x range of domain
            : param Ymin, Ymax : y range of domain
        """
        self.fig = plt.figure()
        self.data = []
        self.times = []
        self.domain = [Xmin, Xmax, Ymin, Ymax ]
        self.cmap = "hot"
        self.Vmax = None
        self.Vmin = None
        # Format for time of snapshot
        self.time_template = "t=%.4f"
        ax = plt.gca()
        self.time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        
    def set_data(self, vals, times =[]):
        """
            : param vals : list of 2d arrays to animate  
            : param times : list of snapshot times
        """
        self.data = vals
        self.times = times

    def set_Vmin_Vmax(self, Vmin, Vmax):
        """
            : param Vmin, Vmax : Density minimum and maximum (None -> auto) 
        """
        self.Vmin = Vmin
        self.Vmax = Vmax

    def set_cmap(self, cmap):
        """ Set color theme  for colour maps:
            https://matplotlib.org/stable/tutorials/colors/colormaps.html
            : param cmap : color theme to use for heat map
        """
        self.cmap = cmap

    def ani_init(self):
        """ Initialises the animation
        """
        data = self.data[0] #np.zeros(self.sizes, dtype=np.float64)
        if self.domain:
            im = plt.imshow(data, cmap=self.cmap, interpolation='nearest',
                            extent=self.domain)
        else:
            im = plt.imshow(data, cmap=self.cmap, interpolation='nearest')
        if  len(self.times) > 0:   
          ti = self.time_text.set_text('')
          return [im, ti]
        return [im] 
    
    def ani_update(self,i):
        """ Update the animation image
            : param i : index of image to update
        """
        data = self.data[i]
        if self.domain:
            im = plt.imshow(data, cmap=self.cmap, vmin=self.Vmin,
                            vmax=self.Vmax, interpolation='nearest',
                            extent=self.domain)
        else:
            im = plt.imshow(data, cmap=self.cmap, vmin=self.Vmin,
                            vmax=self.Vmax, interpolation='nearest')
        if  len(self.times) > 0:
            t = self.times[i]
            ti  = self.time_text.set_text(self.time_template%(t))
            return [im, ti]
        return [im]


    def animate(self,fname=''):
        """ Perform the animation
            If fname is not empty, save the movie in the file
            Else, displays it on the screen 
            : param fname : Output file name if any
        """
        self.anim = animation.FuncAnimation(self.fig, self.ani_update,
                                            init_func=self.ani_init,
                                            frames=len(self.data),
                                            interval=20, blit=False)
        if(fname!= ''):
            self.anim.save(fname, fps=15, writer='ffmpeg')
        else:
            plt.show()

       
# ------------------------------ #
if __name__ == "__main__":
    data = []
    nx = 100
    ny = 100
    x = np.linspace(0,1,nx)
    LN = 20
    x = np.linspace(-3,3,nx)
    y = np.linspace(-3,3,ny)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    # Rotating Gausian hump
    for t in np.linspace(0,1.0,200):
        dx = np.sin(2*np.pi*t)
        dy = np.cos(2*np.pi*t)
        d = np.exp(-((xv-dx)**2+(yv-dy)**2))
        data.append(d)
        
    myani = AnimFct(Xmin=-1,Xmax=1,Ymin=-1,Ymax=1)
    myani.set_data(data)
    myani.set_cmap("cool")
    myani.animate()  

