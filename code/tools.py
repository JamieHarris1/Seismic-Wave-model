import matplotlib.pyplot as plt
from s_wave import SWave

# Some rock properties
granite = [ 5980., 3480., 2.66e3, "b" ]
limestone = [ 5700., 3090., 2.65e3, "y"]
basalt = [ 3380., 1960., 2.15e3, "Blk" ]
shale = [ 2898., 1290, 2.425e3, "m" ]
water = [ 1480., 1., 1e3, "c" ]
PeliticSiltstone = [4580., 2710., 2.34e3, "g"]


def make_detector_signal(sw, Xdet, Xsrc, stype, tmax, model="", fname=""):
    """ Generate detector trace and ray path markers
        : param sw : s_wave instance
        : param Xdet : x position of the detector
        : param Xsrc : x position of the source
        : param stype : type of signal
        : param tmax : maximum time for the graph
        : param model : Model description
        : param fname : filename for figure output (pdf or png file)
    """
    dn = sw.detector_index(Xdet)
    max_val = sw.plot_d(dn, stype)
    dist_d = Xdet-Xsrc
    sw.plot_delay_mark(dist_d,[[0,'P'],[0,'P']], max_val*1.2 , tmax, "blue")
    sw.plot_delay_mark(dist_d,[[0,'P'],[0,'S']], max_val*1.125 , tmax, "cyan")
    sw.plot_delay_mark(dist_d,[[0,'S'],[0,'S']], max_val*1.05 , tmax, "green")
    sw.plot_delay_mark(dist_d,[[0,'P'],[0,'P'],[0,'P'],[0,'P']], max_val*0.975,
                       tmax, "red")
    sw.plot_delay_mark(dist_d,[[0,'S'],[0,'P'],[0,'P'],[0,'P']], max_val*0.825 ,
                       tmax, "salmon")
    sw.plot_delay_mark(dist_d,[[0,'P'],[0,'S'],[0,'P'],[0,'S']], max_val*0.675 ,
                       tmax, "tomato")
    if len(sw.data) > 1:
      sw.plot_delay_mark(dist_d,[[0,'P'],[1,'P'],[1,'P'],[0,'P']], max_val*0.9,
                         tmax, "orange")
      sw.plot_delay_mark(dist_d,[[0,'P'],[1,'S'],[1,'P'],[0,'P']], max_val*0.6,
                         tmax, "lime")
      sw.plot_delay_mark(dist_d,[[0,'S'],[1,'P'],[1,'P'],[0,'P']], max_val*0.75,
                         tmax, "limegreen")
    plt.title("{}: {} at x={}".format(model, stype, Xdet))
    plt.legend()
    if fname:
        plt.savefig(f'graphics/{fname}')
    plt.show()
    plt.cla()
    
def make_detector_reduced_signal(sw, d2, d1, Xdet, Xsrc, stype, tmax, model="",
                              fname=""):
    """ Generate detector trace and ray path markers
        : param sw : s_wave instance
        : param d2 : data for multiple layer model
        : param d1 : data for single layer model (to subtract from d2)
        : param Xdet : x position of the detector
        : param Xsrc : x position of the source
        : param stype : type of signal
        : param tmax : maximum time for the graph
        : param model : Model description
        : param fname : filename for figure output (pdf or png file)
    """
    dn = sw.detector_index(Xdet)
    dist_d = Xdet-Xsrc
    max_val = sw.plot_detector_diff(d2, d1, dn, stype)
    sw.plot_delay_mark(dist_d,[[0,'P'],[0,'P']], max_val*1.2 , tmax, "blue")
    sw.plot_delay_mark(dist_d,[[0,'P'],[0,'S']], max_val*1.125 , tmax, "cyan")
    sw.plot_delay_mark(dist_d,[[0,'S'],[0,'S']], max_val*1.05 , tmax, "green")
    sw.plot_delay_mark(dist_d,[[0,'P'],[0,'P'],[0,'P'],[0,'P']], max_val*0.975 ,
                       tmax, "red")
    sw.plot_delay_mark(dist_d,[[0,'S'],[0,'P'],[0,'P'],[0,'P']], max_val*0.825 ,
                       tmax, "salmon")
    sw.plot_delay_mark(dist_d,[[0,'P'],[0,'S'],[0,'P'],[0,'S']], max_val*0.675 ,
                       tmax, "tomato")
    if len(sw.data) > 1:
      sw.plot_delay_mark(dist_d,[[0,'P'],[1,'P'],[1,'P'],[0,'P']], max_val*0.9,
                       tmax, "orange")
      sw.plot_delay_mark(dist_d,[[0,'P'],[1,'S'],[1,'P'],[0,'P']], max_val*0.6,
                       tmax, "lime")
      sw.plot_delay_mark(dist_d,[[0,'S'],[1,'P'],[1,'P'],[0,'P']], max_val*0.75,
                       tmax, "limegreen")
    plt.title("{}: reduced {} at x={}".format(model, stype, sw.detector_pos(dn)))
    if fname:
        plt.savefig(f'graphics/{fname}')
    plt.show()
    plt.cla()
