import numpy as np
import trimesh

def make_collision_manager(data):
    cm, co = trimesh.collision.scene_to_collision(data["scene"])
    # cm = trimesh.collision.CollisionManager()

    # for well in data["wells"]:
    #     cm.add_object(name=well,mesh=data["wells"][well]["mesh"].mesh)

    data["cm"] = cm
    data["co"] = co

    return data