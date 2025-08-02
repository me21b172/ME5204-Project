import numpy as np

from fea_solver import nr_pipeline
from gmsh_util import create_normal_mesh


def get_boundary_nodes(nodecoords):
    ln = np.where(nodecoords[:,0] == 0)[0]
    rn = np.where(nodecoords[:,0] == np.max(nodecoords[:,0]))[0]
    bn = np.where(nodecoords[:,1] == 0)[0]
    tn = np.where(nodecoords[:,1] == np.max(nodecoords[:,1]))[0]
    return {"ln": ln, "rn": rn, "bn": bn, "tn": tn}

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
    vo = 0  # mm/s, 0 if source ain't moving or no source
    problem_params = {
        "source": {"mode": "absent"}
    }

    boundary_conditions = {
        "top": {
            "mode": "const_flux",
            "q_ext": -1e-3
        },
        "bottom": {
            "mode": "const_flux",
            "q_ext": -1e-3
        },
        "left": {
            "mode": "const_T",
            "T_b": 273+20
        },
        "right": {
            "mode": "const_flux",
            "q_ext": -1e-3
        }
    }
    msf = 3
    nodecoords, ele_con = create_normal_mesh(geo_file='rectangle.geo',
                                             msf_all=msf)
    boundary_nodes = get_boundary_nodes(nodecoords)
    T_init = 273+20
    theta_init = np.zeros((nodecoords.shape[0], 1))+T_init
    temperatures = nr_pipeline(nodecoords, ele_con, boundary_nodes, theta_init, problem_params, 
                               boundary_conditions, props_chooser, mode="static")
    Tmax = np.max(temperatures)-273
    Tmin = np.min(temperatures)-273
    print("T_max is 20 degrees and T_min is around 7 degrees in the paper")
    print(
        f"The calculated maximum temperature is {Tmax:.2f} degrees and minimum temperature is {Tmin:.2f} degrees")
