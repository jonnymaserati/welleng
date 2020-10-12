

def get_SF(surveys, NEVs, cov_NEVs_relative, cov_NEVs_absolute, env, n=0):
    if len(surveys) >= n: # this int can be played with to modify the frequency of the mesh collision check
        reference_NEVs = np.vstack(NEVs)[-n:]
        wm_relative = WellMesh(
            survey=np.vstack(surveys)[-n:],
            NEVs=reference_NEVs,
            cov_NEVs=np.vstack(cov_NEVs_relative)[-n:],
            poss=False
        )

        mesh_relative = wm_relative.make_trimesh()

        # see if there's a collision
        collision_relative = env.cm_donor.in_collision_single(mesh_relative, return_names=True, return_data=True)

        wm_absolute = WellMesh(
                survey=np.vstack(surveys)[-n:],
                NEVs=reference_NEVs,
                cov_NEVs=np.vstack(cov_NEVs_absolute)[-n:],
                poss=False
            )

        mesh_absolute = wm_absolute.make_trimesh()

        # see if there's a collision   
        collision_absolute = env.cm_offset.in_collision_single(mesh_absolute, return_names=True, return_data=True)

        distance_relative = env.cm_donor.min_distance_single(mesh_relative, return_name=True, return_data=True)
        
        # to work out the SF, we need to figure out the closest points on the wellpath to the closest points
        # on the mesh
        closest_point_reference = distance_relative[2].point("__external")
        name_offset_relative = distance_relative[1]
        closest_point_offset = distance_relative[2].point(name_offset_relative)
        reference_NEV = reference_NEVs[spatial.KDTree(reference_NEVs).query(closest_point_reference)[1]]
        offset_NEVs = env.SF_donor.offset_well["NEVs"][np.where(env.SF_donor.offset_well["index"] == env.SF_donor.offset_well["index"][env.SF_donor.offset_well["name"].index(name_offset_relative)])]
        offset_NEV = offset_NEVs[spatial.KDTree(
            offset_NEVs
        ).query(closest_point_offset)[1]]
        distance_CC = norm(offset_NEV - reference_NEV)

        if not collision_relative[0]:
            SF_relative = distance_CC / (distance_CC - distance_relative[0])
        else:
            depth = norm(closest_point_offset - closest_point_reference)
            SF_relative = distance_CC / (distance_CC + depth)
        
        # from SPE-187073-MS equation 1
        # this resolves to the same as the equation above, which is more efficient
        # R_reference = norm(closest_point_reference - reference_NEV)
        # R_offset = norm(closest_point_offset - offset_NEV)
        # SF_relative = (distance_CC) / (R_reference + R_offset)

        SF = SF_relative
        closest_well = name_offset_relative

        
        distance_absolute = env.cm_offset.min_distance_single(mesh_absolute, return_name=True, return_data=True)

        # to work out the SF, we need to figure out the closest points on the wellpath to the closest points
        # on the mesh
        closest_point_reference = distance_absolute[2].point("__external")
        name_offset_absolute = distance_absolute[1]
        closest_point_offset = distance_absolute[2].point(name_offset_absolute)
        reference_NEV = reference_NEVs[spatial.KDTree(reference_NEVs).query(closest_point_reference)[1]]
        offset_NEVs = env.SF_offset.offset_well["NEVs"][np.where(env.SF_offset.offset_well["name"] == name_offset_absolute)]
        offset_NEV = offset_NEVs[spatial.KDTree(
            offset_NEVs
        ).query(closest_point_offset)[1]]
        distance_CC = np.linalg.norm(offset_NEV - reference_NEV)
        SF_absolute = distance_CC / (distance_CC - distance_absolute[0])

        if not collision_absolute[0]:
            SF_absolute = distance_CC / (distance_CC - distance_absolute[0])
        else:
            depth = norm(closest_point_offset - closest_point_reference)
            SF_absolute = distance_CC / (distance_CC + depth)

        # from SPE-187073-MS equation 1
        # this resolves to the same as the equation above, which is more efficient
        # R_reference = norm(closest_point_reference - reference_NEV)
        # R_offset = norm(closest_point_offset - offset_NEV)
        # SF_absolute = (distance_CC) / (R_reference + R_offset)

        if SF_absolute < SF:
            SF = SF_absolute
            closest_well = name_offset_absolute
    
    else:
        collision_relative = (False, {}, [])
        collision_absolute = (False, {}, [])
        SF = 1.0
        closest_well = None

    return (SF, closest_well, collision_relative, collision_absolute)