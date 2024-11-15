import gmsh
import meshio

# Parameters
geo_file = "rectange.geo"  # Path to the GEO file
mesh_filename = 'rectangle.msh'
meshFactorForEntireSurface = 3
meshFactorForPatchSurface = 0.35
surfaceID = [3]
# Initialize GMSH
gmsh.initialize()
gmsh.open(geo_file)

# Set the global mesh size factor to 3 (for planes other than Plane 3)
gmsh.option.setNumber("Mesh.MeshSizeFactor", meshFactorForEntireSurface)

# Create a mesh field to apply a different mesh size to Plane 3
gmsh.model.mesh.field.add("Constant", 1)  # Add a constant field with ID 1
gmsh.model.mesh.field.setNumber(1, "VIn", meshFactorForPatchSurface)  # Set mesh size factor for Plane 3 to 1

# Assuming Plane 3 has ID 3; if not, replace 3 with the actual ID
gmsh.model.mesh.field.setNumbers(1, "SurfacesList", surfaceID)

# Set this field as the background mesh to apply it specifically to Plane 3
gmsh.model.mesh.field.setAsBackgroundMesh(1)

# Generate the mesh
gmsh.model.mesh.generate(2)

# Write to a .msh file
gmsh.write(mesh_filename)

# Finalize GMSH
gmsh.finalize()

# Read the mesh using MeshIO
mesh = meshio.read(mesh_filename)

print("Mesh successfully generated and written to", mesh_filename)