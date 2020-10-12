import numpy as np
from numpy import radians, sin, cos, sqrt, pi

from scipy.spatial import KDTree

import trimesh

class WellMesh:

    def __init__(
        self,
        survey,
        NEVs,
        cov,
        cov_HLAs=False,
        N_verts=12,
        sigma=3.0,
        wellbore_radius=0.15,
        surface_margin=0,
        degrees=True,
        poss=True,
    ):

        self.survey = self._assert_shape(survey)
        if degrees:
            self.survey[:,-2:] = radians(self.survey[:,-2:])

        NEVs = self._assert_shape(NEVs)
        if poss:
            x, y, z = NEVs.T
            self.NEVs = np.array(y, x, z).T
        else:
            self.NEVs = NEVs

        # assert cov_NEVs.shape[-2:] == (3,3) and cov_NEVs.shape[-3] > 1, "cov_NEVs must be an (-1,3,3) shape array"
        if cov_HLAs:
            self.cov_NEVs = None
            self.cov_HLAs = cov.T
        
        else:
            self.cov_NEVs = cov

            # get the cov_HLAs: assume that cov_NEVs shape is (-1,3,3)
            # cov_HLAs returned shape is (3,3,-1) so this needs to be transposed
            self.cov_HLAs = self._NEV_to_HLA(self.survey, self.cov_NEVs.T)

        self.N_verts = int(N_verts)
        self.sigma = sigma
        self.wellbore_radius = wellbore_radius
        self.surface_margin = surface_margin

        self.ellipses = self._get_ellipses()
        self.align_verts()

        self.faces = self.get_faces()
        self.make_trimesh()       

    ### Helper functions ###

    def get_faces(self):
        step = self.N_verts
        vertices = []
        edges = []
        faces = []
        total_verts = len(self.ellipses.reshape(-1,3))
        rows = int(total_verts / self.N_verts)
        
        # make first end        
        B = np.arange(1, step - 1, 1)
        C = np.arange(2, step, 1)
        A = np.zeros_like(B)
        temp = np.array([A, B, C]).T
        faces.extend(temp.tolist())
            
        # make cylinder
        temp = [np.array([step, 0, step - 1, 2 * step - 1, step, step - 1])]
        for row in range(0, rows - 1):
            verts = row * self.N_verts

            A_start = verts
            B_start = verts + step
            C_start = verts + step + 1
            D_start = verts + 1
            
            A_stop = verts + step - 1
            B_stop = verts + step + step - 1
            C_stop = verts + step + step
            D_stop = verts + step

            A = np.arange(A_start, A_stop)
            B = np.arange(B_start, B_stop)
            C = np.arange(C_start, C_stop)
            D = np.arange(D_start, D_stop)

            temp.extend(np.array([C, D, A, B, C, A]).T)

            # last = np.array([C_stop - 1, D_stop - 1, A_start, B_start, C_stop - 1, A_start])
            last = np.array([B_start, A_start, D_stop - 1, C_stop - 1, B_start, D_stop -1])

            temp.extend([last])

            # temp = np.vstack((temp, np.array([C, D, A, B, C, A]).T, last))
        
        faces.extend(np.stack(temp, axis=0).reshape(-1,3).tolist())

        # first = np.array([step, 0, step - 1, 2 * step - 1, step, step - 1])

        # A = np.arange(0, total_verts - step - 1)
        # B = np.arange(step, total_verts - 1)
        # C = np.arange(step + 1, total_verts)
        # D = np.arange(1, total_verts - step)

        # last = np.array([total_verts - step, total_verts - 2 * step, total_verts - step - 1, total_verts - 1, total_verts - step, total_verts - step - 1])
        
        # temp = np.vstack((first, np.array([C, D, A, B, C, A]).T, last))
        # faces.extend(temp.reshape(-1,3).tolist())
            
        # make final end     
        B = np.arange(total_verts - step + 1, (total_verts - 1), 1)
        C = np.arange(total_verts - step + 2, (total_verts), 1)
        A = np.full_like(B, total_verts - step)
        temp = np.array([A, B, C]).T
        faces.extend(temp.tolist())

        # return np.array(faces).flatten()
        return faces


    def _get_ellipses(
        self,
        ):

        H = sqrt(np.absolute(self.cov_HLAs[0,0])) * self.sigma + self.wellbore_radius + self.surface_margin
        L = sqrt(np.absolute(self.cov_HLAs[1,1])) * self.sigma + self.wellbore_radius + self.surface_margin
        A = sqrt(np.absolute(self.cov_HLAs[2,2])) * self.sigma + self.wellbore_radius + self.surface_margin

        lam = np.linspace(0, 2 * pi, self.N_verts, endpoint=False)
        theta = np.zeros_like(lam)

        # x = [h * cos(theta) * cos(lam) for h in H]
        # y = [l * cos(theta) * sin(lam) for l in L]
        # z = [a * sin(theta) for a in A]

        x = H.reshape(-1,1) * cos(theta) * cos(lam)
        y = L.reshape(-1,1) * cos(theta) * sin(lam)
        z = A.reshape(-1,1) * sin(theta)
        
        # ellipses = np.array([np.array([x.flatten(), y.flatten(), z.flatten()]).T for x, y, z, list(zip(x, y, z))])

        ellipses = np.stack((x, y, z), axis=-1)
        
        ellipses = self._HLA_to_NEV(self.survey, ellipses, cov=False)
        
        ellipses = np.vstack([ellipse + nev for ellipse, nev in list(zip(ellipses, self.NEVs))]).reshape(ellipses.shape)

        return ellipses

    def _NEV_to_HLA(
        self, survey, NEV, cov=True
        ):

        trans = self._transform(survey)

        if cov:
            HLAs = [
                np.dot(np.dot(mat, NEV.T[i]), mat.T) for i, mat in enumerate(trans.T)
            ]
            
            HLAs = np.vstack(HLAs).reshape(-1,3,3).T
            
        else:
            HLAs = [
                np.dot(NEV, mat.T) for i, mat in enumerate(trans.T)
            ]
            
        return HLAs        
    

    def _HLA_to_NEV(
        self, survey, HLA, cov=True
        ):

        trans = self._transform(survey)

        if cov:
            NEVs = [
                np.dot(np.dot(mat.T, HLA.T[i]), mat) for i, mat in enumerate(trans.T)
            ]
        
            NEVs = np.vstack(NEVs).reshape(-1,3,3).T
        
        else:
            NEVs = [
                np.dot(hla, mat) for hla, mat in list(zip(HLA, trans.T))
            ]
            
        return np.vstack(NEVs).reshape(HLA.shape)


    def _transform(
        self, survey
        ):

        inc = np.array(survey[:,1])
        azi = np.array(survey[:,2])
        
        return np.array([
            [cos(inc) * cos(azi), -sin(azi), sin(inc) * cos(azi)],
            [cos(inc) * sin(azi), cos(azi), sin(inc) * sin(azi)],
            [-sin(inc), np.zeros_like(inc), cos(inc)]
        ])


    def _assert_shape(
        self, arr
        ):
        
        # assert arr.shape[-1] == 3 and arr.shape[-2] > 1, "Must be an array or list of 2 or more 3d arrays"
        return np.copy(np.array(arr).reshape(-1,3))


    def get_ellipse_vectors(self, vertices, pos_center):
        vectors = vertices - pos_center
        normals = np.linalg.norm(vectors, axis=-1)
        unit_vectors = vectors / normals.reshape(-1,1)
    
        return (vectors, normals, unit_vectors)


    def align_verts(self):
        """
        The meshing gets unstable when the inclination is close to vertical (since high side
        can quickly shift azimuth). This code cycles through the ellipses and makes sure that
        they're all lined up relative to each other.

        """
        verts_new_list = []
        for i, verts in enumerate(self.ellipses):
            if i == 0:
                verts_new_list.append([verts])
                continue
            else:
                vecs_1, norms_1, units_1 = self.get_ellipse_vectors(verts_new_list[-1][0], self.NEVs[i - 1])
                vecs_2, norms_2, units_2 = self.get_ellipse_vectors(self.ellipses[i], self.NEVs[i])

                nearest = KDTree(units_2)

                key = units_1[0]
                idx_key = nearest.query(key)[1]

                rotation = units_1[3]
                idx_rotation = nearest.query(rotation)[1]

                # re-order vecs_2 to align with vecs_1
                verts_new = np.vstack((verts[idx_key:], verts[:idx_key]))

                if (idx_rotation - idx_key) < 0 and abs(idx_rotation - idx_key) < len(verts) / 2:
                    verts_new = verts_new[::-1]

                verts_new_list.append([verts_new])
        
                # if self.verbose:
                #     print("#########################################")
                #     print(f"units_1 = \n{units_1}")
                #     print(f"units_2 = \n{units_2}")
                #     print(f"{idx_key = }\n{idx_rotation = }")
            
        self.ellipses = np.vstack(verts_new_list).reshape(self.ellipses.shape)

        return self.ellipses
        # return np.array(verts_new)

    def make_trimesh(self):
        self.mesh = trimesh.Trimesh(
            vertices=self.ellipses.reshape(-1,3),
            # faces=self.faces.reshape(-1,3)
            faces=self.faces,
            process=True
        )

        if len(trimesh.repair.broken_faces(self.mesh)) > 0:
            trimesh.repair.fill_holes(self.mesh)

        return self.mesh