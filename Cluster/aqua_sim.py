import numpy as np
from fea_solver import nr_pipeline
from gmsh_util import create_normal_mesh, plot_mesh

def Q(point, source, ro):
    if source is None:
        return 0
    x = point[0,0] - source[0,0]
    y = point[0,1] - source[0,1]
    Qo = 440/(np.pi*0.75*0.75)  ## amplitude in W/mm^2 
    return Qo*np.exp(-(x**2+y**2)/ro**2)  ## W/mm^3
# we consider only one phase - alpha
def rho_Ti(T):
    return (-5.13e-5*(T**2)-0.01935*T+4451)/1e9

def cp_Ti(T, process='heating'):
    lin = 0.25*T+483
    heat = 13000*np.exp(-0.5*(((T-1160)/90)**2))/(90*np.sqrt(2*np.pi))
    cool = 13000*np.exp(-0.5*(((T-952)/90)**2))/(90*np.sqrt(2*np.pi))
    if process == 'heating':
        return lin+heat
    elif process == 'cooling':
        return lin+cool

def k_Ti(T):
    return (0.012*T+3.3)/1e3

def props_chooser(T, process):
    return rho_Ti(T), cp_Ti(T, process), k_Ti(T)

if __name__ == '__main__':
    ro = 0.75 #mm
    vo = 10 #mm/s
    problem_params = {
        "ro" : ro, 
        "vo" : vo,
        "Q"  : Q,
        "source_present" : True
    }
    h = 20e-6 #W/(mm^2)
    T_inf = 273+150
    boundary_conditions = {
        "top": {
            "mode" : "convection",
            "h"    : h,
            "T_inf": T_inf
        },
        "bottom": {
            "mode" : "convection",
            "h"    : h,
            "T_inf": T_inf
        },
        "left": {
            "mode" : "convection",
            "h"    : h,
            "T_inf": T_inf
        },
        "right": {
            "mode" : "convection",
            "h"    : h,
            "T_inf": T_inf
        },
    }
    converged_msf = 3
    converged_msf_adapt = 1.5
    converged_dt = 0.05
    T_init = 150+273  # preheat
    t_final = 4
    nodecoords, ele_con = create_normal_mesh(
        geo_file='rectangle_small.geo',
        msf_all=0.1)
    # plot_mesh(nodecoords, ele_con)
    n_nodes = nodecoords.shape[0]
    theta_init = np.zeros((n_nodes, 1))+T_init
    theta_init = np.zeros((nodecoords.shape[0], 1))+T_init
    temperatures = nr_pipeline(nodecoords, ele_con, theta_init, problem_params, boundary_conditions, 
                               props_chooser, dt=converged_dt, t_final=t_final, source=np.array([[40, 1]]), mode="transient")

