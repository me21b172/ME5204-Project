import gmsh

def createMesh(geo_file,msf_all,side = None,msf_adapt=None,x_s=None,y_s=None,is_adapt=False):
    gmsh.initialize()
    gmsh.open(geo_file)
    if is_adapt:
        gmsh.model.mesh.MeshSizeExtendFromBoundary = 0
        gmsh.model.mesh.MeshSizeFromPoints = 0
        gmsh.model.mesh.MeshSizeFromCurvature = 0
        gmsh.model.mesh.field.add("Box", 1)
        gmsh.model.mesh.field.setNumber(1, "VOut", msf_all)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.field.setNumber(1, "VIn", msf_adapt)     
        gmsh.model.mesh.field.setNumber(1, "XMax", min(x_s + side,100)) 
        gmsh.model.mesh.field.setNumber(1, "XMin", max(x_s - side,0)) 
        gmsh.model.mesh.field.setNumber(1, "YMax", min(y_s + side,50)) 
        gmsh.model.mesh.field.setNumber(1, "YMin", max(y_s - side,0))  
        # Apply the combined field as a background mesh
        gmsh.model.mesh.field.setAsBackgroundMesh(1)
    else:
        gmsh.option.setNumber("Mesh.MeshSizeFactor", msf_all)
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