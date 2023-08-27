import numpy as np
import matplotlib.pyplot as plt
import code.animate_map as am
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Pde2dFast:
  """ A class to solve dv/dt=F(t,v) where F is a differential
      operator acting on the 3 dimensional array 'v'.
      Shape of v : Nx, Ny, Nf
  """

  def __init__(self, v0, Xmin=0, Xmax=1, Ymin=0, Ymax=1):
    """
        : param v0         : initial condition as an array or a list
        : param Lx         : Size of x domain
        : param Ly         : Size of y domain
    """
    self.reset(v0, Xmin, Xmax, Ymin, Ymax)
    

  def reset(self, v0, Xmin=0, Xmax=1, Ymin=0, Ymax=1, t0=0):
    """ Set the initial parameters.
        : param v0         : initial condition
        : param L          : length of domain
        : param dt         : integration time step
        : param t0         : initla time
    """
    self.t = t0 # initial time
    self.v = np.array(v0, dtype='float64') # ensure we use floats!
    self.Nx = v0.shape[0] # total number of points
    self.Ny = v0.shape[1] # total number of points
    self.Xmin = Xmin        # 
    self.Xmax = Xmax        # 
    self.Ymin = Ymin        # 
    self.Ymax = Ymax        # 
    self.Lx = (Xmax-Xmin)   # domain size
    self.Ly = (Ymax-Ymin)   # domain size
    self.dx = self.Lx/(self.Nx-1) # lattice spacing in x
    self.dy = self.Ly/(self.Ny-1) # lattice spacing in y
    # domain used for 2d plots 
    self.domain = [self.Xmin, self.Xmax, self.Ymin, self.Ymax]
    
    self.n = 0 # time iteration number
    self.l_t = []  # list of t values for plots
    self.l_f = []  # list of array f values (arrays) for figures
    #self.boundary(self.v) # make sure the boundary condition is set
    
  def F(self, v):
    """ Returns update for v as an array except for the first and last points 
        To be implemented in subclass boundary.
        See the example below the class definition
    
        : param v : current function value. Array of Shape Nx, Ny, Nf
        : return  : right hand side of equation for v 
    """
    pass

  def boundary(self, v):
    """ Enforce the boundary conditions. To be implemented in subclass.
        : param v : current function value. Array of Shape Nx, Ny, Nf
    """
    pass
  
  def RK4_one_step(self):
      """ Perform a single integration step of the 4th order Runge Kutta method
      """

      k1 = self.F(self.t, self.v)
      K = self.v+0.5*self.dt*k1
      self.boundary(K)

      k2 = self.F(self.t+0.5*self.dt, K)
      K = self.v+0.5*self.dt*k2
      self.boundary(K)

      k3 = self.F(self.t+0.5*self.dt, K)
      K = self.v+self.dt*k3
      self.boundary(K)
 
      k4 = self.F(self.t+self.dt, K)

      # self.v -> v(t+dt)
      self.v += self.dt/6.0*(k1+2.0*(k2+k3)+k4)      
      self.boundary(self.v)
      self.t += self.dt;

  def extra_data(self):
    """ Called each time data are saved for figures
        To use in sub classes to generate more data
    """
    return
      
  def iterate(self, tmax, dt, fig_dt=-1, extra_dt = -1):
    #relax_RK4(self, err, dt=0.1,nmax=-1,n_fig=0):
    """ Relax until largest update is smaller than err.
      : param tmax   : iterate until tmax.
      : param dt     : integration time step.
      : param fig_dt : time step between figure data.
      : param extra_dt : time step between extra data.
   """
    self.dt = dt

    if(fig_dt < 0) : fig_dt = self.dt*0.99 # save all data
    if(extra_dt < 0) : extra_dt = self.dt*0.99 # save all data

    next_fig_t = fig_dt 
    next_extra_t = extra_dt 

    self.l_f.append(np.array(self.v)) # save inital condition
    self.l_t.append(np.array(self.t)) # save inital condition
    self.extra_data()

    while(self.t < tmax):
      self.RK4_one_step()
      
      if(self.t >= next_fig_t): # save fig when next_fig_t is reached
        self.l_f.append(np.array(self.v))
        self.l_t.append(self.t)
        next_fig_t += fig_dt # set the next figure time
      if(self.t >= next_extra_t):
        self.extra_data()
        next_extra_t += extra_dt # set the next figure time
      
    # set plot data
    self.l_f.append(np.array(self.v))
    self.l_t.append(self.t)
    self.extra_data()

  def snapshots(self, d, fig_dt=-1, fname_prefix="", vmin=None, vmax=None,
                fname_suffix=".pdf"):
    """
        Generate snpshots of the saved data at regular intervals
        : param d : index of function to plot
        : param fig_dt : time interval between snapshots 
        : param fname_prefix : filename prefix for output. "" -> show on screeen
        : param vmin : Lower bound value (None -> auto)
        : param vmax : Upper bound value (None -> auto)
        : param fname_suffix : file type : ".pdf", ".png" ...
    """
    t = self.l_t[0]
    next_t = t- 1e-12

    for i in range(len(self.l_t)):
      t = self.l_t[i]
      if fig_dt<0 or t > next_t:
        plt.clf()
        ax = plt.gca()

        a = self.l_f[i]
        data = np.array(a[:,:,int(d)]).squeeze().transpose()
        if self.domain:
          im = plt.imshow(data, cmap='hot',vmin=vmin, vmax=vmax,
                          interpolation='nearest', extent=self.domain)
        else:
          im = plt.imshow(data, cmap='hot',vmin=vmin, vmax=vmax,
                          interpolation='nearest')
        next_t += fig_dt

        # Color bar on the right hand side. Same size as figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        # time of snapshot
        ax.text(0.05, 0.9, "t=%.4f"%(t), transform=ax.transAxes)
        
        if fname_prefix != "":
          fname = "{}_{}{}".format(fname_prefix, i, fname_suffix)
          plt.savefig(f'graphics/{fname}')
        else:  
          plt.show()

  def heat_map(self, d, fname="", vmin=None, vmax=None):
    """
        Generate snpshots of the selected fields in v
        : param d : index of function to plot
        : param fig_dt : time interval between snapshots 
        : param fname : filename for output. "" -> show on screeen
        : param vmin : Lower bound value (None -> auto)
        : param vmax : Upper bound value (None -> auto)
    """
    ax = plt.gca()
    data = np.array(self.v[:,:,d]).squeeze().transpose()
    if self.domain:
      im = plt.imshow(data, cmap='hot',vmin=vmin, vmax=vmax,
                      interpolation='nearest', extent=self.domain)
    else:
      im = plt.imshow(data, cmap='hot',vmin=vmin, vmax=vmax,
                      interpolation='nearest')
      divider = make_axes_locatable(ax)
      cax = divider.append_axes("right", size="5%", pad=0.05)
      plt.colorbar(im, cax=cax)

    if fname != "":
      plt.savefig(f'graphics/{fname}')
    plt.show()


  def animation(self, Vmin, Vmax, d, fname=""):
     """
         Generate animation of the stored data.
        : param Vmin : Lower bound value (None -> auto)
        : param Vmax : Upper bound value (None -> auto)
        : param d : index of function to animate
        : param fname : input filename  ("" -> show on screen)
     """
     anim = am.AnimFct(0, self.Xmax,0, self.Zmax)
     anim.set_Vmin_Vmax(Vmin, Vmax)

     data = []
     times = []
     print("size l_f=",len(self.l_f))
     for i in range(len(self.l_f)):
       a = self.l_f[i]
       #print("a:", a.shape)
       cpa = np.array(a[:,:,int(d)]).squeeze().transpose()
       data.append(cpa)
       times.append(self.l_t[i])
     anim.set_data(data, times)
     anim.set_cmap('hot')
     anim.animate(fname)

  def pos_to_index(self, pos):
    """ Compute the grid index position of (x,y) 
        : param pos : list, tuple  or array for (x, y)
        : return : grid coordinates as tuple
    """
    i = round((pos[0]-self.Xmin)/self.dx)
    j = round((pos[1]-self.Ymin)/self.dy)
    return i,j
  
# Tests follow; only run when not importing the module.

if __name__ == "__main__":

  class diff(PDE_2D_fast):
    """ A class to relax to static solutions of the diffusion equation. """
    def F(self, t, v):
       """ Diffusion equation in 2 D
         : param t : current time
         : param v : current function
       """       
       eq = np.zeros(self.Np)
       # eval equation except for end points for which eq=0
       eq[1:self.Np-1] = (v[2:self.Np] + v[:self.Np-2] -2*v[1:self.Np-1])/self.dx**2
       return(eq)
   
    def boundary(self, v):
       """ Enforce the boundary conditions. 
           Left: 1 (large pool). Right 0 (empty pool)
       """
       v[0] = 1
       v[-1] = 0

    def best_dt(self, coef):
       """ Best dt for this equation
       : param coef : adjustment coeficient
       """
       return(coef*self.dx**2)
    
  Np=50
  diff_eq = diff(np.zeros([Np]), 1)
  diff_eq.iterate(1, diff_eq.best_dt(0.5), fig_dt=0.1)
  #print("No of iterations: ",n) 
  for nf in range(len(diff_eq.l_f)):
    #print(diff_eq.l_t[nf])
    diff_eq.plot(nf)
    plt.show()
