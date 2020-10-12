import numpy as np
import trimesh, open3d

def make_trimesh_scene(data):
    """
    Construct a trimesh scene. A collision manager can't be saved, but a scene can and a scene can be imported
    into a collision manager.
    """
    scene = trimesh.scene.scene.Scene()
    for well in data["wells"]:
        mesh = data["wells"][well]["mesh"].mesh
        scene.add_geometry(mesh, node_name=well, geom_name=well, parent_node_name=None)

    data["scene"] = scene
    
    return data


def transform_trimesh_scene(scene, origin=None, scale=100, redux=0.25):
    """
    Transforms a scene by scaling it, reseting the origin/datum and performing
    a reduction in the number of triangles to reduce the file size.

    Params:
        scene: trimesh.scene
            A trimesh scene of well meshes
        origin: 3d array [x, y, z]
            The origin of the scene from which the new scene will reset to [0, 0, 0]
        scale: float
            A scalar reduction will be performed using this float
        redux: float
            The desired reduction ratio for the number of triangles in each mesh

    """
    i = 0
    if not origin:
        # probably should do thi and just leave as [0, 0, 0], but this works for Grane
        T = np.array([0, 0, 0])
    else:
        T = np.array(origin)
    scene_transformed = trimesh.scene.scene.Scene()
    for well in scene.geometry.items():
        name, mesh = well
        mesh_new = mesh.copy()

        ### reduce the triangle count using open3d algo to reduce the file size ###
        mesh_temp = open3d.geometry.TriangleMesh()
        mesh_temp.vertices=open3d.utility.Vector3dVector(np.asarray(mesh_new.vertices))
        mesh_temp.triangles=open3d.utility.Vector3iVector(np.asarray(mesh_new.faces))
        mesh_temp = mesh_temp.simplify_quadric_decimation(int(len(mesh_temp.triangles) * redux))

        mesh_new.vertices = np.asarray(mesh_temp.vertices)
        mesh_new.faces = np.asarray(mesh_temp.triangles)
        
        # localize the coorindate system
        mesh_new.vertices -= T

        ### change axis convention for visualisation ###
        x, y, z = mesh_new.vertices.T
        mesh_new.vertices = (np.array([x, z * -1, y]) / scale).T
        scene_transformed.add_geometry(mesh_new, node_name=name, geom_name=name)

    return scene_transformed