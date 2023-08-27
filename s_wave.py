import numpy as np
import math
from code.s_wave_base import SWaveBase


def snell(theta1, V1, V2):
    '''
        Implementation for Snell's Law returning the new theta,
        the angle of reflection/refraction
        
        : param theta1        : Incidence angle
        : param V1            : Speed before medium
        : param V2            : Speed after medium
        : return              : Angle of reflection/refraction
    '''    
    #Return Theta2 using Snell's Law
    return math.asin(V2 / V1 * math.sin(theta1))

  
class SWave(SWaveBase):
    """ 
        A class to solve the seismic wave equation in an inhomogeneous
        environment.
    """

    def __init__(self, Xmax, Zmax, Nx, Gamma=0.2, n_abs=30):
        """
        :  param Xmax       : X domain is [0, Xmax]
        :  param Zmax       : Z domain is [0, Zmax]
        :  param Nx         : Node number in X direction
        :  param Gamma      : Maximum damping value
        :  param n_abs      : Width of damping border (an integer)
        """
        super().__init__(Xmax, Zmax, Nx, Gamma, n_abs)
        
    def F(self, t, v_):
        """
            Equation for seismic waves in inhomogeneous  media
            
            : param t : Current time
            : param v_ : Current function value (a vector of shape [Nx, Nz, 4])
            : return  : Right hand side of the equation for v_]
        """
        # Initial exitation
        e = self.excite(t)
        if len(e) > 0:
            v_[self.excite_pos[0], self.excite_pos[1], :] = e
        
        #Initialise empty eq arrray, same size as v_
        eq = np.zeros([self.Nx, self.Nz, 4])

        #v_ is 3D arrays of shape [Nx, Nz, 4], sperate out into components
        v = v_[:, :, 0]
        w = v_[:, :, 1]
        
        #Find nnumber of columns for v and w
        n = v.shape[1]
        m = w.shape[1]
        
        #Scan enitre grid except edge nodes
        #Using discretised relations for derrivatives
        #Roll of n or m (depending on v/w) adjusts row, +-1 adjusts column
        dv_xx = ((np.roll(v, 1*n) + np.roll(v, -1*n) -2*v) / self.dx**2)
        dv_zz = ((np.roll(v, 1) + np.roll(v, -1) -2*v)/ self.dx**2)
        dv_xz = ((np.roll(v, 1*n + 1) + np.roll(v, -1*n - 1) 
            - np.roll(v, -1*n + 1) - np.roll(v, 1*n - 1)) / (4*self.dx*self.dz))
        
        dw_xx = ((np.roll(w, 1*m) + np.roll(w, -1*m) -2*w) / self.dx**2)
        dw_zz = ((np.roll(w, 1) + np.roll(w, -1) -2*w) / (self.dx**2))
        dw_xz = ((np.roll(w, 1*m + 1 ) + np.roll(w, -1*m - 1) 
            - np.roll(w, -1*m + 1) - np.roll(w, 1*m - 1)) / (4*self.dx*self.dz))
        
        #Set up our 4 first order equaitons in time
        #Updates eq with v'=dv/dt w'=dw/dt
        eq[:, :, 0] =  v_[:, :, 2]
        eq[:, :, 1] =  v_[:, :, 3]
        
        #Updates eq for dv'/dt, dw'/dt using rearrangement of wave equation
        eq[:, :, 2] = ((self.lam + self.mu)*(dv_xx + dw_xz) +
                        self.mu * (dv_xx + dv_zz)) / self.rho
        eq[:, :, 3] = ((self.lam + self.mu)*(dw_zz + dv_xz) +
                        self.mu * (dw_xx + dw_zz)) / self.rho
                
        #Ensures edge nodes still equal to 0
        for i in range(4):
            eq[:, :, i] = np.pad(eq[1:-1, 1:-1, i], pad_width=((1, 1), (1, 1)), 
                                mode = 'constant', constant_values = 0)
        return(eq) 

    def boundary(self, v_):
        """
            Function to enforce boundary conditions on sides and floor of medium
            
            : param v_ : Current function value (a vector of shape [Nx, Nz, 4])
        """
        
        #Enforce discrete boundary conditions
        #Adjust v for all z=0 excluding edges
        v_[:, 0, 0][1:self.Nx - 1] = (v_[:, 1, 0] + (np.roll(v_[:, 1, 1], -1) 
            - np.roll(v_[:, 1, 1], 1)) * 0.5)[1:- 1]
        
        #Adjust w for all z=0 excluding edges
        v_[:, 0, 1][1:self.Nx - 1] = (v_[:, 1, 1] + 0.5 * self.lam[:, 1] /\
            (self.lam[:, 1] + 2*self.mu[:, 1]) * (np.roll(v_[:, 1, 0], -1)
            - np.roll(v_[:, 1, 0], 1)))[1:- 1]
        
        #Adjust v and w time derrivatives at edges
        v_[:, :, 2] *= 1 - self.Gamma
        v_[:, :, 3] *= 1 - self.Gamma

        
    def dist_offset(self, theta, path):
        """
            Function to find the horizontal distance travelled by a wave path 
            and the time taken
            
            : param theta     : Intital incidence angle of path
            : param path      : A list of type [ [l1, W1], [l2, W2], ...]
                where li is the layer index and Wi the type of wave: "P" or "S".
            : return dx       : Horizontal distance travelled by a wave path
            : return t        : Time taken to travel horizontal distance dx
        """
        dx=0
        t=0
        #Intialise array of dictionaries to represent our 3 layer model
        l1, l2, l3 = {}, {}, {}
        layers = [l1, l2, l3]
        
        #Set dictionary values wihtin each layer to corresponding ones from self.data
        for i in range(len(self.data)):
            layers[i] = {
                "Layer" : i,
                "Type" : self.data[i][4],
                "Distance" : self.data[i][0],
                "Density" : self.data[i][3],
                "VP" : self.data[i][1],
                "VS" : self.data[i][2]
                }
        
        
        

        for p in range(len(path)):
            layer, wtype = path[p]
            
            #Initialise values considering current layer and wave type 
            V = layers[layer]["V" + wtype]
            Di = layers[layer]["Distance"]
            
            
            #calculating x offset
            dx += Di * math.tan(theta)
            
            #calculating time delay
            t += Di / (V * math.cos(theta))
            
            #calculate new theta  incidence angle after relfection/refraction
            try:
                nextwtype = path[p+1][1]
                Vnext = layers[int(path[p+1][0][-1])]["V" + nextwtype]
                theta = math.asin(V / Vnext * math.sin(theta))
            except:
                pass
            
        return dx, t 


    def angle_and_delay(self, dx, path, err = 0.1):
        """ 
            Function to determine the incidence angle needed to achieve 
            a desired horizontal distance and the time taken for a specified path
            
            : param dx        : Desired Horizontal distance
            : param path      : A list of type [ [l1, W1], [l2, W2], ...]
                where li is the layer index and Wi the type of wave: "P" or "S".
            : param err       : The acceptable margin of error in distance
            : return theta    : Angle needed to achieve horizontal distance x
            : return x        : Actual Horizontal distance achieved
            : return t        : Time taken to cover horizontal distance x
        """
        
        lowerb, theta, x = 0, 0, 0
        upperb = math.pi / 2
        
        #Iterate until dx is sufficiently close to x
        while abs(dx - x) > err:
            #Calculate new horizontal distance using updated theta
            x = self.dist_offset(theta, path)[0]
            
            #Use bisection method to adjust lower\upper bound
            if x > dx:
                upperb = theta
                
            elif x < dx:
                lowerb = theta
            theta = (lowerb + upperb) / 2
        
        x, t = self.dist_offset(theta, path)
        return theta, x ,t

        
    def eval_detector_diff(self, d1, d2, dn, data_type ):  
        """ 
            Function to compute the difference between two lists of detector 
            signals d1 and d2 employing 3 different methods of difference
            
            : param d1        : A list of lists of the type [ti, lvi, lwi] and so
                lvi[dn] will be the v displacement at the detector dn at time ti.
            : param d2        : A list of lists of the type [ti, lvi, lwi] and so
                lvi[dn] will be the v displacement at the detector dn at time ti.
            : param dn        : The index of the detector
            : param data_type : Select how the difference must be computed, 
                will be one of "v", "w" or "Mod" or "Phase"
            : return          : Array of type [[t0, x0], [t1, x1], .. [tn, xn]]
                where xi are the difference of displacement between d2 and d1 at detector dn
        """
        #Constuct data into numpy arrays 
        d1_t, d1_lv, d1_lw = list(zip(*d1))
        d1_t, d1_lv, d1_lw =np.array(d1_t), np.array(d1_lv), np.array(d1_lw)
        d2_t, d2_lv, d2_lw = list(zip(*d2))
        d2_t, d2_lv, d2_lw =np.array(d2_t), np.array(d2_lv), np.array(d2_lw)
        
        #Only want dn detecotor information
        d1_lv, d1_lw = d1_lv[:, dn], d1_lw[:, dn]
        d2_lv, d2_lw = d2_lv[:, dn], d2_lw[:, dn]
        
        #Find dispalcement difference
        v = d2_lv - d1_lv
        w = d2_lw - d1_lw
        
        #Select how the difference must be computed
        if data_type == 'v':
            xis = v
            
        elif data_type == 'w':
            xis = w
            
        elif data_type == 'Mod':
            xis = np.sqrt(v**2 + w**2)


        elif data_type == 'Phase':
            xis = np.atan2(w, v)
            
        #Return array of xis in form of [[t0,x0],[t1,x1]...,[tn,xn]]
        return np.array(list(zip(d1_t, xis)))

    

