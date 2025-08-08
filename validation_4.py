import numpy as np
import matplotlib.pyplot as plt
from fea_solver import nr_pipeline
from gmsh_util import create_normal_mesh, interpolate, find_mesh_size

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

def heat_transfer_fem(temperatures, nodecoords, ele_con, boundary_nodes):
    q = 0
    for ele in ele_con:
        econ = ele-1
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
        Ts = np.array([T_1, T_2, T_3])
        dT_physical = Jac_inv @ np.array([T_2-T_1, T_3-T_1]).reshape(2, -1)
        for l,m in zip([0,1,2],[1,2,0]):
            n1 = econ[l]
            n2 = econ[m]
            d12 = np.linalg.norm(nodecoords[n1, :2]-nodecoords[n2, :2])
            if (n1 in boundary_nodes["tn"]) and (n2 in boundary_nodes["tn"]):
                q += k((Ts[l]+Ts[m])/2)*dT_physical[1]*d12*1000 # 1m thick chimney => 1000 mm
    total_heat_transfer = 8*q
    return total_heat_transfer

def heat_transfer_nlc(temperatures, hin, Tin, boundary_nodes):
    T_avg = np.mean(temperatures[boundary_nodes["tn"]])
    Q_nl = hin*(Tin-T_avg)*50*1000*8 #0.05 m wide, 1m thick, 8times
    return Q_nl 

def plot_comparison(temperatures, nodecoords, ele_con, ax, ms, start=False):
    T_notes = [97.1, 88.9, 73.2, 46.6, 163.3, 151.8, 107.8, 263.1, 232.4]
    T_book = [60, 55, 40, 23, 152, 138, 89, 273, 256]
    locations = [[0, 0], [50, 0], [100, 0], [150, 0], [0, 50], [50, 50], [100, 50], [0, 100], [50, 100]]
    locations = np.concatenate([np.array(locations), np.zeros((len(T_notes), 1))], axis=-1)
    
    T_fem = interpolate(locations, temperatures, nodecoords, ele_con)-273.15
    
    x = np.arange(len(T_book))
    if start:
        ax.plot(x, T_notes, 'ro-', label='Notes', linewidth=2, markersize=6) #convection
        ax.plot(x, T_book, 'bo-', label='Book', linewidth=2, markersize=6)  #convection and radition

    ax.plot(x, T_fem, 'o--', label=f'FEM (mesh_size={ms:.2f}mm)', linewidth=2, markersize=6)
    



    
    
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
    T_init = Tout

    fig, ax = plt.subplots()
    q_fems = []
    q_nlcs = []
    
    msfs = [25, 20, 15, 10, 5, 1, 0.5]
    mesh_sizes = []
    for i, msf in enumerate(msfs): 
        nodecoords, ele_con = create_normal_mesh(
                                geo_file='trapezium.geo',
                                msf_all=msf
                                )
        mesh_sizes.append(find_mesh_size(nodecoords, ele_con))
        boundary_nodes = get_boundary_nodes(nodecoords)
        theta_init = np.zeros((nodecoords.shape[0], 1))+T_init
        temperatures = nr_pipeline(nodecoords, ele_con, boundary_nodes, theta_init, 
                                problem_params, boundary_conditions, 
                                props_chooser, mode="static")
        start = i==0
        plot_comparison(temperatures, nodecoords, ele_con, ax, mesh_sizes[-1], start)
        q_fems.append(heat_transfer_fem(temperatures, nodecoords, ele_con, boundary_nodes))
        q_nlcs.append(heat_transfer_nlc(temperatures, hin, Tin, boundary_nodes))
    ax.set_title("Comparison of Notes, Book and Custom Temperatures")
    ax.set_xlabel("Measurement Point Index")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True)
    ax.legend(fontsize=8)
    # plt.show()
    plt.savefig("Temperature Comparison")
    
    ax.cla()
    
    x = np.arange(len(q_fems))
    sample = np.zeros(len(msfs))
    ax.semilogx(mesh_sizes, sample+2775, 'o--', label='Notes_FEM', linewidth=2, markersize=6)
    ax.semilogx(mesh_sizes, sample+1463, 'o--', label='Notes_NLC', linewidth=2, markersize=6)
    ax.semilogx(mesh_sizes, sample+1291, 'o--', label='Book_FDM(convection)', linewidth=2, markersize=6)
    ax.semilogx(mesh_sizes, sample+1993, 's--', label='Book_FDM(all)', linewidth=2, markersize=6)
    ax.semilogx(mesh_sizes, sample+1994, '^--', label='Book_NLC', linewidth=2, markersize=6)
    ax.semilogx(mesh_sizes, q_fems, 'o-', label='FEM', linewidth=2, markersize=6)
    ax.semilogx(mesh_sizes, q_nlcs, 'o-', label='Newton Cooling', linewidth=2, markersize=6)

    ax.set_xticks(mesh_sizes, [str(round(mesh_size,1)) for mesh_size in mesh_sizes])
    ax.tick_params(axis='x', which='minor', labelbottom=False)
    ax.tick_params(axis='x', which='major', labelsize=7)
    ax.set_title("Comparison of Heat transfer")
    ax.set_xlabel("Mesh size(mm)")
    ax.set_ylabel("Heat transfer (W)")
    ax.grid(True, which='both')
    ax.legend(fontsize=8)
    # plt.show()
    plt.savefig("Heat transfer comparison")
    

