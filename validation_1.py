import numpy as np

from fea_solver import nr_pipeline
from gmsh_util import create_normal_mesh

def get_boundary_nodes(nodecoords):
    ln = np.where(nodecoords[:,0] == 0)[0]
    rn = np.where(nodecoords[:,0] == np.max(nodecoords[:,0]))[0]
    bn = np.where(nodecoords[:,1] == 0)[0]
    tn = np.where(nodecoords[:,1] == np.max(nodecoords[:,1]))[0]
    return {"ln": ln, "rn": rn, "bn": bn, "tn": tn}

def Q(point, centre, ro):
    x = point[0, 0]
    return 15e-3*(x/100)*(1-x/100)  # W/mm^3


def rho_T(T):
    return 7e-6*np.ones_like(T)  # kg/mm^3


def cp_T(T, process='heating'):
    return 465*np.ones_like(T)   # J/kg.K


def k_T(T):
    return (100+0.004*(T-50-273)**2)*1e-3  # W/mmK


def props_chooser(T, process):
    return rho_T(T), cp_T(T, process), k_T(T)


if __name__ == '__main__':
    problem_params = {
        "source": {"mode": "volumetric", "Q": Q}
    }

    boundary_conditions = {
        "top": {
            "mode": "const_flux",
            "q_ext": 0
        },
        "bottom": {
            "mode": "const_T",
            "T_b": 273+100
        },
        "left": {
            "mode": "const_flux",
            "q_ext": 0
        },
        "right": {
            "mode": "const_flux",
            "q_ext": 0
        }
    }
    msf = 3
    nodecoords, ele_con = create_normal_mesh(geo_file='square.geo',
                                             msf_all=msf)

    boundary_nodes = get_boundary_nodes(nodecoords)
    T_init = 273+50
    theta_init = np.zeros((nodecoords.shape[0], 1))+T_init
    temperatures = nr_pipeline(nodecoords, ele_con, boundary_nodes, theta_init, problem_params, 
                               boundary_conditions, props_chooser, dt=1, t_final=9,
                               mode="transient")

    Ta = temperatures[np.where((nodecoords[:, 0] == 0)
                              * (nodecoords[:, 1] == 100))[0][0], -1]-273
    Tb = temperatures[np.where((nodecoords[:, 0] == 100)
                              * (nodecoords[:, 1] == 100))[0][0], -1]-273
    print("Both nodes have temperatures of 54.746K in the reference paper")
    print(
        f"The calculated temperature at node a is {Ta:.2f}K and at node b is {Tb:.2f}K")
