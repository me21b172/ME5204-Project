import numpy as np

from fea_solver import nr_pipeline
from gmsh_util import create_normal_mesh, create_box_mesh, plot_distribution

sigma = 5.67e-2  # W/mm^2K^4, Stefan-Boltzmann constant

def get_boundary_nodes(nodecoords):
    ln = np.where(nodecoords[:,0] == 0)[0]
    rn = np.where(nodecoords[:,0] == np.max(nodecoords[:,0]))[0]
    bn = np.where(nodecoords[:,1] == 0)[0]
    tn = np.where(nodecoords[:,1] == np.max(nodecoords[:,1]))[0]
    return {"ln": ln, "rn": rn, "bn": bn, "tn": tn}

def Q(point,temp ,source, ro):
    if source is None:
        return 0
    x = point[0,0] - source[0,0]
    y = point[0,1] - source[0,1]
    Qo = 5 ## amplitude in W/mm^2 
    return Qo*np.exp(-(x**2+y**2)/ro**2) + sigma*(temp - 273)**4  ## W/mm^3


def rho_T(T):
    return 7.6e-6*np.ones_like(T)  # kg/mm^3


def cp_T(T, process='heating'):
    return 658*np.ones_like(T)   # J/kg.K


def k_T(T):
    return 25e-3*np.ones_like(T)  # W/mmK


def props_chooser(T, process):
    return rho_T(T), cp_T(T, process), k_T(T)


if __name__ == '__main__':
    ro = 2  # mm
    vo = 2  # mm/s, 0 if source ain't moving or no source
    problem_params = {
        "source": {"mode": "laser", "vo": vo, "ro": ro, "Q": Q, "position": np.array([[100, 25]])}
    }


    boundary_conditions = {
        "top": {
            "mode": "const_flux",
            "q_ext": 1e-3
        },
        "bottom": {
            "mode": "const_flux",
            "q_ext": 1e-3
        },
        "left": {
            "mode": "const_T",
            "T_b": 273+20
        },
        "right": {
            "mode": "const_flux",
            "q_ext": 1e-3
        }
    }
    # nodecoords, ele_con = create_box_mesh(
    #                         geo_file='rectangle.geo',
    #                         msf_all=3,
    #                         msf_adapt=1.5,
    #                         length=100,
    #                         width=8,
    #                         x_s=50,
    #                         y_s=25)
    nodecoords, ele_con = create_normal_mesh(
                            geo_file='rectangle_small.geo',
                            msf_all=1
                            )
    boundary_nodes = get_boundary_nodes(nodecoords)
    T_init = 273+20
    theta_init = np.zeros((nodecoords.shape[0], 1))+T_init
    temperatures = nr_pipeline(nodecoords, ele_con, boundary_nodes, theta_init, problem_params, boundary_conditions, 
                               props_chooser, t_final=25, mode="transient")

    plot_distribution(temperatures[:, -1], temperatures[:, -1].min(), temperatures[:, -1].max(), nodecoords, ele_con, is_node=True)
    Tmax = temperatures[:, -1].max()-273
    Tmin = temperatures[:, -1].min()-273
    print("T_max is around 607 degrees and T_min is around 20 degrees in the paper") #transient analysis
    print(
        f"The calculated maximum temperature is {Tmax:.2f} degrees and minimum temperature is {Tmin:.2f} degrees")
