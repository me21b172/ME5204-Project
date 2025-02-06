import numpy as np
from transient_util import nr_iterative_rect
from gmsh_util import create_box_mesh

if __name__ == '__main__':
    converged_msf = 3
    converged_msf_adapt = 1.5
    converged_dt = 0.05
    T_l = 20+273
    # t_final = 50
    t_final = 0.1
    nodecoords,ele_con = create_box_mesh(geo_file='rectangle.geo',
                                    msf_all=converged_msf,
                                    msf_adapt = converged_msf_adapt,
                                    length = 100,
                                    width = 8,
                                    x_s = 50,
                                    y_s = 25) 
    theta_init = np.zeros((nodecoords.shape[0],1))+T_l
    temperatures,phases = nr_iterative_rect(nodecoords,ele_con,np.array([[100,25]],dtype = np.float32),theta_init,dt = converged_dt,\
                                    t_final =t_final+1e-3,type = "transient",mode = "phase_change")
    
    np.save('T.npy',temperatures)
    np.save('P.npy',phases)
    