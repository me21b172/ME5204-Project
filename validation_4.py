import numpy as np
from fea_solver import nr_pipeline
from gmsh_util import create_normal_mesh, plot_mesh, plot_boundary_nodes, plot_distribution, interpolate

def get_boundary_nodes(nodecoords):
    ln = np.where(nodecoords[:,0] == 0)[0]
    rn = np.where(np.isclose(nodecoords[:,0] + nodecoords[:,1], 150, rtol=0, atol=1e-3))[0]
    bn = np.where(nodecoords[:,1] == 0)[0]
    tn = np.where(nodecoords[:,1] == np.max(nodecoords[:,1]))[0]
    return {"ln": ln, "rn": rn, "bn": bn, "tn": tn}

def rho(T):
    return 2200e-9*np.ones_like(T)  #kg/mm^3

def cp(T, process='heating'):
    return 1425*np.ones_like(T) #J/kgK

def k(T):
    return 1.4e-3*np.ones_like(T)  #W/mmK

def props_chooser(T, process):
    return rho(T), cp(T, process), k(T)

if __name__ == '__main__':
    problem_params = {
        "source": {"mode": "absent"}
    }
    hin = 70e-6 #W/(mm^2K)
    Tin = 273+300 #K
    
    hout = 21e-6 #W/(mm^2K)
    Tout = 273+20 #K
    
    boundary_conditions = {
        "top": {
            "mode" : "convection",
            "h"    : hin,
            "T_inf": Tin
        },
        "bottom": {
            "mode" : "convection",
            "h"    : hout,
            "T_inf": Tout
        },
        "left": {
            "mode" : "const_flux",
            "q_ext"    : 0
        },
        "right": {
            "mode" : "const_flux",
            "q_ext"    : 0
        }
    }
    nodecoords, ele_con = create_normal_mesh(
                            geo_file='trapezium.geo',
                            msf_all=25
                            )
    boundary_nodes = get_boundary_nodes(nodecoords)
    # plot_mesh(nodecoords, ele_con)
    # plot_boundary_nodes(nodecoords, boundary_nodes)


    T_init = Tout
    theta_init = np.zeros((nodecoords.shape[0], 1))+T_init
    temperatures = nr_pipeline(nodecoords, ele_con, boundary_nodes, theta_init, 
                               problem_params, boundary_conditions, 
                               props_chooser, mode="static")

    print("At 0, 100")
    print(interpolate(np.array([[0, 100, 0]]), temperatures, nodecoords, ele_con)-273)
    
    print("At 50, 100")
    print(interpolate(np.array([[50, 100, 0]]), temperatures, nodecoords, ele_con)-273)
    # plot_distribution(temperatures-273.15, nodecoords, ele_con, is_node=True)
    
    term = 0
    for ele in ele_con:
        econ = ele-1
        nnode = econ.shape[0]
        boundary = nodecoords[np.ix_(econ, [0, 1])]
        dN = np.array([[-1, 1, 0], [-1, 0, 1]])
        Jac = np.matmul(dN, boundary)
        if np.linalg.det(Jac) < 0:
            # reordering for the direction to be counter clockwise
            econ[0], econ[1] = econ[1], econ[0]
            boundary = nodecoords[np.ix_(econ, [0, 1])]
            Jac = np.matmul(dN, boundary)
        Jac_inv = np.linalg.inv(Jac)
        T_1, T_2, T_3= temperatures[econ]
        dT_physical = Jac_inv @ np.array([T_2-T_1, T_3-T_1]).reshape(2, -1)
        for l,m in zip([0,1,2],[1,2,0]):
            n1 = econ[l]
            n2 = econ[m]
            d12 = np.linalg.norm(nodecoords[n1, :2]-nodecoords[n2, :2])
            if (n1 in boundary_nodes["tn"]) and (n2 in boundary_nodes["tn"]):
                term += d12*dT_physical[1]
                
    Q_fem = (k(Tout)*term*1000)*8 #1m to 1000 mm
    print(f"Heat transfer from Finite element integration: {Q_fem} W")
    
    T_avg = np.mean(temperatures[boundary_nodes["tn"]])
    Q_nl = hin*(Tin-T_avg)*50*1000*8
    print(f"Heat transfer from Newton's law of cooling integration: {Q_nl} W")
    

