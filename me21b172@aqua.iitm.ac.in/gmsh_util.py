import gmsh
import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_distribution(variable,mini,maxi,nodecoords, ele_con,is_node = True):
    '''
    variable : variable which should be plotted
    mini : minimum possible value of the variable 
    maxi : maximimum possible value of the variable
    nodecoords,ele_con : mesh data
    is_node = True if shape of variable is nnodes x 1, False if shape is nele x 1
    The phases are 0 - alpha, 1 - beta, 2 - liquid
    '''
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[30, 1], wspace=0.05)

    ax = fig.add_subplot(gs[0, 0])
    if(is_node == True):
        variable_ele = (variable[ele_con[:, 0] - 1] + variable[ele_con[:, 1] - 1] + variable[ele_con[:, 2] - 1]) / 3
    else:
        variable_ele = variable
    cmap = plt.get_cmap('jet')
    normalized_values = (variable_ele - mini) / (maxi - mini)
    colors = cmap(normalized_values)
    for elei, coli in zip(ele_con, colors):
        econ = elei - 1
        x_data = nodecoords[econ, 0]
        y_data = nodecoords[econ, 1]

        ax.fill(x_data, y_data, color=coli)
        ax.plot(list(x_data) + [x_data[0]], list(y_data) + [y_data[0]], color=(0.25,0.25,0.25),linewidth = 0.5)
    ax.axis('equal')
    
    # Colorbar area
    cax = fig.add_subplot(gs[0, 1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=mini, vmax=maxi))
    sm.set_array([])
    plt.colorbar(sm, cax=cax, shrink = 0.5)
    plt.show()


def plot_mesh(nodecoords,ele_con):
    plt.figure(1)

    for elei in ele_con:
        econ = elei - 1
        x_data = nodecoords[econ,0]
        y_data = nodecoords[econ,1]

        plt.plot(list(x_data)+[x_data[0]],list(y_data)+[y_data[0]],'r')
    plt.axis('equal')
    plt.show()

def find_triangle_params(rep,nodecoords,ele_con):
    rep_x,rep_y,rep_z = rep[0,0],rep[0,1],rep[0,2]
    all_x,all_y,all_z = nodecoords[:,0],nodecoords[:,1],nodecoords[:,2]
    # params = np.zeros((ele_con.shape[0],2))
    for i,elei in enumerate(ele_con):
        econ = elei-1
        boundary = nodecoords[np.ix_(econ,[0,1])]
        dN = np.array([[-1,1,0],[-1,0,1]])
        Jac = np.matmul(dN,boundary)
        if np.linalg.det(Jac)<0:
            econ[0],econ[1] = econ[1],econ[0] #reordering for the direction to be counter clockwise
            boundary = nodecoords[np.ix_(econ,[0,1])] 
            Jac = np.matmul(dN,boundary)

        triangle = Path(boundary)
        if triangle.contains_point(np.array([rep_x,rep_y]),radius = 1e-9):
            v1,v2,v3 = econ[0],econ[1],econ[2]
            A = np.array([[all_x[v2]-all_x[v1],all_x[v3]-all_x[v1]],[all_y[v2]-all_y[v1],all_y[v3]-all_y[v1]]])
            B = np.array([[rep_x-all_x[v1]],[rep_y-all_y[v1]]])
            params = (np.linalg.inv(A)@B)
            return econ,params[0,0],params[1,0]
    print("Triangle not found")
    return -1

def create_normal_mesh(geo_file,msf_all,side = None,msf_adapt=None,x_s=None,y_s=None,is_adapt=False):
    gmsh.initialize()
    gmsh.open(geo_file)
    gmsh.option.setNumber("Mesh.MeshSizeFactor", msf_all)
    gmsh.model.mesh.generate(2)
    mesh_filename = 'reqd_mesh.msh'
    gmsh.write(mesh_filename)
    gmsh.finalize()
    return read_mesh(mesh_filename)

def create_box_mesh(geo_file,msf_all,msf_adapt,length,width,x_s,y_s):
    gmsh.initialize()
    gmsh.open(geo_file)
    gmsh.model.mesh.MeshSizeExtendFromBoundary = 0
    gmsh.model.mesh.MeshSizeFromPoints = 0
    gmsh.model.mesh.MeshSizeFromCurvature = 0
    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VOut", msf_all)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.model.mesh.field.setNumber(1, "VIn", msf_adapt)     
    gmsh.model.mesh.field.setNumber(1, "XMax", min(x_s + length/2,100)) 
    gmsh.model.mesh.field.setNumber(1, "XMin", max(x_s - length/2,0)) 
    gmsh.model.mesh.field.setNumber(1, "YMax", min(y_s + width/2,50)) 
    gmsh.model.mesh.field.setNumber(1, "YMin", max(y_s - width/2,0))  
    # Apply the combined field as a background mesh
    gmsh.model.mesh.field.setAsBackgroundMesh(1)
    gmsh.model.mesh.generate(2)
    mesh_filename = 'reqd_mesh.msh'
    gmsh.write(mesh_filename)
    gmsh.finalize()
    return read_mesh(mesh_filename)


def read_mesh(filepath):
    '''
    Takes in an msh file and should return nodal coordinates and element connectivity in each physical group along with the boundary nodes 
    '''
    gmsh.initialize()
    gmsh.open(filepath)
    print(f"Reading {filepath}")
    print(f"Number of nodes in the mesh: {int(gmsh.option.getNumber('Mesh.NbNodes'))}")
    print(f"Number of triangles in the mesh: {int(gmsh.option.getNumber('Mesh.NbTriangles'))}\n")

    #Get all nodes
    dim = -1
    tag = -1
    nodeTags, nodecoords, _ = gmsh.model.mesh.getNodes(dim,tag)
    nodecoords = nodecoords.reshape(-1,3) #tags start from 1

    #Get all triangles
    eleType = 2
    tag = -1
    elements_t,ele_con = gmsh.model.mesh.getElementsByType(eleType,-1)
    ele_con = ele_con.reshape(-1,3)  #tags start from 1

    gmsh.finalize()
    return [nodecoords,ele_con] 