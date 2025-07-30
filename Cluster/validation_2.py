import numpy as np

from fea_solver import nr_pipeline
from gmsh_util import create_normal_mesh


def Q(point, source, ro):
    if source is None:
        return 0
    x = point[0,0] - source[0,0]
    y = point[0,1] - source[0,1]
    Qo = 5 ## amplitude in W/mm^2 
    return Qo*np.exp(-(x**2+y**2)/ro**2)  ## W/mm^3


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
        "ro": ro,
        "vo": vo,
        "source_present": False,
        "Q": Q
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
    T_init = 273+20
    theta_init = np.zeros((nodecoords.shape[0], 1))+T_init
    temperatures = nr_pipeline(nodecoords, ele_con, theta_init, problem_params, 
                               boundary_conditions, props_chooser, mode="static")
    Tmax = np.max(temperatures)-273
    Tmin = np.min(temperatures)-273
    print("T_max is 20 degrees and T_min is around 7 degrees in the paper")
    print(
        f"The calculated maximum temperature is {Tmax:.2f} degrees and minimum temperature is {Tmin:.2f} degrees")
