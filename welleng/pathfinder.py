"""
This was a huge help in generating the code below:
https://www.redblobgames.com/pathfinding/a-star/implementation.html
"""

import numpy as np
import collections
# import networkx as nx
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import numpy as np
import heapq
import random
import math
import welleng as we
import vedo
from welleng import connector, utils
from tqdm import tqdm
import ray
import trimesh
import sys
import psutil
import time
import copy
from graph_tool.all import *
import graph_tool as gt


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return not self.elements

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)


class Queue:
    def __init__(self):
        self.elements = collections.deque()

    def empty(self) -> bool:
        return not self.elements

    def put(self, x):
        self.elements.append(x)

    def get(self):
        return self.elements.popleft()


def heuristic(current_position, target_position):
    """
    Parameters
    ----------
        current_position: (1, 3) array of floats
            The current position in nev coordinates.
        target_position: (1, 3) array of floats
            The target position in nev coordinates.
    """
    return np.sum(
        np.absolute(
            np.array(current_position) - np.array(target_position)
        )
    )


def get_connector_nodes(octnode_1, octnode_2, G_node_1):
    n1 = connector.Node(
        pos=octnode_1.position,
        md=0.,
        inc=G_node_1['inc'],
        azi=G_node_1['azi']
    )
    n2 = connector.Node(
        pos=octnode_2.position
    )
    return (n1, n2)


def connect_graph_nodes(node1, node2, dls_design, force_min_curve=False):
    n1, n2 = [
        connector.Node(
            pos=n['pos'],
            vec=n['vec'],
        ) for n in [node1, node2]
    ]
    n1.md = node1['md']
    if force_min_curve:
        n2.md = node2['md']
        n2.pos, n2.pos_nev, n2.pos_xyz = None, None, None

    c = connector.Connector(
        n1, n2, dls_design=dls_design, min_error=1e-4, min_tangent=0.,
        force_min_curve=force_min_curve,
        delta_radius=50,
    )
    return c


def get_cost(n1, n2, dls_max):
    c = connector.Connector(
        node1=n1,
        node2=n2,
    )
    dls = (
        (30 * 180)
        / (min(c.radius_critical, c.radius_critical2) * np.pi)
    )
    if dls > dls_max:
        return False

    inc, azi = np.degrees(utils.get_angles(
        c.vec_target, nev=True
    ).reshape(2))

    return (c.md_target, inc, azi)


# def get_cost(n1, n2, dls_max):
#     vec = n2.pos_nev - n1.pos_nev
#     cost = np.linalg.norm(vec)
#     inc, azi = np.degrees(utils.get_angles(
#         vec / cost, nev=True
#     ).reshape(2))

#     return (cost, inc, azi)


@ray.remote
class CollisionManager:
    def __init__(self, scene, constraints={}):
        self.cm = trimesh.collision.scene_to_collision(scene)[0]
        self.constraints = constraints

    @ray.method(num_returns=1)
    def in_collision_single(self, mesh):
        data = self.cm.in_collision_single(
            mesh.mesh, return_names=True, return_data=True
        )
        if data[0] is False:
            return False
        else:
            if len(data[1]) == 1 and "shale" in data[1]:
                # get the contact points
                mesh_contacts = [
                    p.point for p in data[2]
                    if 'shale' in p.names
                ]
                # get the closest points on the well path
                k = KDTree(np.array([mesh.s.n, mesh.s.e, mesh.s.tvd]).T)
                # get the indexes of well path points
                survey_contacts = np.unique(k.query(mesh_contacts)[1])
                incs = mesh.s.inc_deg[min(survey_contacts):max(survey_contacts) + 1]
                azis = mesh.s.azi_grid_deg[min(survey_contacts):max(survey_contacts) + 1]
                bools = []
                for inc, azi in zip(incs, azis):
                    if inc < 30:
                        bools.append(False)
                    elif (
                        inc > 30
                        and (
                            45 <= azi <= 135
                            or 225 <= azi <= 315
                        )
                    ):
                        bools.append(False)
                    else:
                        bools.append(True)
                return all(bools)
            else:
                return True


@ray.remote
class Graph:
    def __init__(self):
        self.graph = gt.Graph(directed=True)
        # self.counter = 0

        # define node (vertex) property maps
        node_md = self.graph.new_vertex_property("double")
        node_inc = self.graph.new_vertex_property("double")
        node_azi = self.graph.new_vertex_property("double")
        node_pos = self.graph.new_vertex_property("vector<double>")
        node_vec = self.graph.new_vertex_property("vector<double>")
        node_priority = self.graph.new_vertex_property("double")

        # map to internal properties
        self.graph.vp.md = node_md
        self.graph.vp.inc = node_inc
        self.graph.vp.azi = node_azi
        self.graph.vp.pos = node_pos
        self.graph.vp.vec = node_vec
        self.graph.vp.priority = node_priority

        # define edge propert maps
        edge_weight = self.graph.new_edge_property("double")
        edge_md = self.graph.new_edge_property("vector<double>")
        edge_inc = self.graph.new_edge_property("vector<double>")
        edge_azi = self.graph.new_edge_property("vector<double>")
        edge_mesh = self.graph.new_edge_property("object")

        # map to internal properties
        self.graph.ep.weight = edge_weight
        self.graph.ep.md = edge_md
        self.graph.ep.inc = edge_inc
        self.graph.ep.azi = edge_azi
        self.graph.ep.mesh = edge_mesh

    # def count(self):
    #     self.counter += 1
    #     return self.counter - 1

    # @ray.method(num_returns=1)
    # def get_counter(self):
    #     return self.counter - 1

    @ray.method(num_returns=1)
    def get_graph(self):
        return self.graph

    def set_attributes(self, attributes, property='vp'):
        function = {
            'vp': self.graph.vp,
            'ep': self.graph.ep
        }
        for i in attributes:
            if property == 'ep':
                n1, n2 = i
                n = self.graph.edge(n1, n2)
            else:
                n = i
            for k, v in attributes[i].items():
                if property == 'vp':
                    v = np.nan if v is None else v
                function[property][k][n] = v

    @ray.method(num_returns=1)
    def add_node(
        self, pos, inc=None, azi=None, vec=None, md=None, priority=np.inf
    ):
        # without the final param there's a risk the node will be selected as
        # current while the md hasn't been verified yet.
        # idx = self.count()
        n = self.graph.add_vertex()

        if vec is not None:
            inc, azi = np.degrees(we.utils.get_angles(
                vec, nev=True
            ).reshape(2))
        elif inc is not None and azi is not None:
            vec = we.utils.get_vec(
                inc=inc,
                azi=azi,
                nev=True,
                deg=True
            ).reshape(3)

        attributes = {n: {
            'pos': pos, 'vec': vec, 'inc': inc, 'azi': azi,
            'md': md, 'priority': priority
        }}
        self.set_attributes(attributes)

        return self.graph.vertex_index[n]

    def update_node(self, n, pos, vec, md):
        inc, azi = np.degrees(we.utils.get_angles(
            vec, nev=True
        ).reshape(2))
        attributes = {n: {
            'pos': pos, 'vec': vec, 'inc': inc, 'azi': azi,
            'md': md
        }}
        self.set_attributes(attributes)


    @ray.method(num_returns=1)
    def get_node(self, idx):
        n = self.graph.vertex(idx)
        node = {
            'pos': self.graph.vertex_properties['pos'][n],
            'vec': self.graph.vertex_properties['vec'][n],
            'md': self.graph.vertex_properties['md'][n],
            'inc': self.graph.vertex_properties['inc'][n],
            'azi': self.graph.vertex_properties['azi'][n],
            'priority': self.graph.vertex_properties['priority'][n]
        }
        return node

    @ray.method(num_returns=1)
    def get_number_of_active_nodes(self):
        active = len(np.where(
            self.graph.vertex_properties['priority'].get_array() != np.inf
        )[0])
        return active

    @ray.method(num_returns=1)
    def get_best_well(self, source=0, target=1):
        shortest = gt.topology.shortest_distance(
            self.graph, source=source, target=target,
            weights=self.graph.ep.weight, dag=True
        )
        return shortest

    @ray.method(num_returns=1)
    def get_best_node(self):
        arr = self.graph.vertex_properties['priority'].get_array()
        best = arr.argmin()
        return best

    def deactivate_node(self, n):
        attributes = {n: {
            'priority': np.inf
        }}
        self.set_attributes(attributes)
        self.graph.clear_vertex(n)

    def decay_node(self, n, decay):
        self.graph.vp['priority'][n] += decay

    def add_edge(
        self, u, v, survey, weight=np.inf, mesh=None,
        md=None, inc=None, azi=None
    ):
        e = self.graph.add_edge(
            self.graph.vertex(u), self.graph.vertex(v)
        )
        attributes = {
            (u, v): {
                'weight': weight,
                'mesh': mesh,
                'md': survey.md,
                'inc': survey.inc_rad,
                'azi': survey.azi_grid_rad
            }
        }
        self.set_attributes(attributes, property='ep')

    def remove_edge(self, u, v):
        e = self.graph.edge(u, v)
        self.graph.remove_edge(e)

    # @ray.method(num_returns=1)
    # def get_edge_index(self, u, v):
    #     e = self.graph.edge(u, v)

    #     return e



# @ray.remote  # (num_cpus=3)
# class Graph:
#     # TODO rewrite with e.g. graph-tool to see if there's a significant
#     # performance boost.
#     def __init__(self):
#         """
#         Let's do this in NEV
#         """
#         self.graph = nx.DiGraph()
#         self.counter = 0

#         # if sys.platform == 'linux':
#         #     psutil.Process().cpu_affinity([0])

#     def count(self):
#         self.counter += 1
#         return self.counter - 1

#     @ray.method(num_returns=1)
#     def get_counter(self):
#         return self.counter - 1

#     @ray.method(num_returns=1)
#     def get_graph(self):
#         return self.graph

#     @ray.method(num_returns=1)
#     def add_node(
#         self, pos, inc=None, azi=None, vec=None, md=None
#     ):
#         # without the final param there's a risk the node will be selected as
#         # current while the md hasn't been verified yet.
#         idx = self.count()
#         self.graph.add_node(idx)

#         if vec is not None:
#             inc, azi = np.degrees(we.utils.get_angles(
#                 vec, nev=True
#             ).reshape(2))
#         elif inc is not None and azi is not None:
#             vec = we.utils.get_vec(
#                 inc=inc,
#                 azi=azi,
#                 nev=True,
#                 deg=True
#             ).reshape(3)

#         attributes = {idx: {
#             'pos': pos, 'vec': vec, 'inc': inc, 'azi': azi, 'md': md,
#             'priority': np.inf
#         }}
#         nx.set_node_attributes(self.graph, attributes)

#         return idx

#     def update_node(self, idx, pos, vec, md):
#         inc, azi, vec = self.get_angles_and_vec(inc=None, azi=None, vec=vec)
#         attributes = {idx: {
#             'pos': pos, 'vec': vec, 'inc': inc, 'azi': azi, 'md': md,
#             'priority': np.inf
#         }}
#         nx.set_node_attributes(self.graph, attributes)

#     @staticmethod
#     def get_angles_and_vec(inc, azi, vec):
#         if vec is not None:
#             inc, azi = np.degrees(we.utils.get_angles(
#                 vec, nev=True
#             ).reshape(2))
#         elif inc is not None and azi is not None:
#             vec = we.utils.get_vec(
#                 inc=inc,
#                 azi=azi,
#                 nev=True,
#                 deg=True
#             ).reshape(3)

#         return (inc, azi, vec)

#     def add_edge(self, n1, n2, weight, mesh=None, survey=None):
#         # self.graph.add_edge(n1, n2, weight=weight, mesh=mesh)
#         self.graph.add_edge(
#             n1, n2,
#             weight=weight,
#             md=survey.md,
#             inc=survey.inc_rad,
#             azi=survey.azi_grid_rad,
#             mesh=mesh
#         )

#     def remove_edge(self, n1, n2):
#         self.graph.remove_edge(n1, n2)

#     @ray.method(num_returns=1)
#     def get_node(self, idx):
#         return self.graph._node[idx]

#     def remove_node(self, idx):
#         self.graph.remove_node(idx)
#         # del self.graph._node[idx]
#         # self.graph.remove_node(idx)
#         # self.counter -= 1

#     @ray.method(num_returns=1)
#     def number_of_nodes(self):
#         return len(self.graph._node)

#     @ray.method(num_returns=1)
#     def get_current(self):
#         return min(
#             [
#                 node for node in self.graph._node
#                 # if self.graph._node[node]['final'] is True
#             ],
#             key=(lambda x: self.graph._node[x]['priority'])
#         )

    # @ray.method(num_returns=1)
    # def get_priority(self, idx, decay=None, set_value=None):
    #     if bool(set_value):
    #         self.graph._node[idx]['priority'] = set_value
    #     if bool(decay):
    #         self.graph._node[idx]['priority'] += decay
    #     return self.graph._node[idx]['priority']

    # def update_edge_attributes(self, n1, n2, attributes):
    #     nx.set_edge_attributes(
    #         self.graph, {(n1, n2): attributes}
    #     )
    #     # for k, v in attributes.items():
    #     #     assert k in self.graph[n1][n2], "Invalid attribute"
    #     #     self.graph[n1][n2][k] = v

    # @ray.method(num_returns=1)
    # def get_graph_data(self):
    #     return self.__dict__

    # @ray.method(num_returns=1)
    # def get_survey(self, n1, n2):
    #     # path = single_source_dijkstra(
    #     #     self.graph,
    #     #     n1,
    #     #     n2
    #     # )
    #     path = [
    #         p for p in nx.all_simple_paths(self.graph, 0, n2)
    #         # if n1 in p
    #         # and n2 in p
    #         if all((n1 in p, n2 in p))
    #     ]
    #     temp = []
    #     for (u, v) in zip(path[0][:-1], path[0][1:]):
    #         data = self.graph.get_edge_data(u, v)
    #         temp.append([data['md'], data['inc'], data['azi']])

    #     return np.hstack(temp)

    # @ray.method(num_returns=1)
    # def get_edge_data(self, u, v):
    #     return self.graph.get_edge_data(u, v)


@ray.remote(num_cpus=0)
class Frames:
    def __init__(self):
        # if sys.platform == 'linux':
        #     psutil.Process().cpu_affinity([0])

        self.generation = []
        self.N = []
        self.best_wells = []
        self.best_nodes = []
        self.meshes = []
        self.edges = []
        self.graph = None

    def add_frame(self, generation, best_node, best_well):
        self.generation.append(generation)
        self.N.append(self.get_number_of_frames())
        self.best_nodes.append(best_node)
        self.best_wells.append(best_well)

    @ray.method(num_returns=1)
    def get_number_of_frames(self):
        return len(self.meshes)

    def add_mesh(self, mesh):
        self.meshes.append(mesh)

    def add_edge(self, edge):
        self.edges.append(edge)

    @ray.method(num_returns=1)
    def get_data(self):
        return self.__dict__

    @ray.method(num_returns=1)
    def get_best_well(self):
        if bool(self.best_wells):
            return self.best_wells[-1]
        else:
            return None


class Object:
    def __init__(self, **obj_dict):
        self.__dict__.update(obj_dict)


def rrt_star_search(
    octree,
    start_pos, start_inc, start_azi, target_pos, start_md=0,
    cm=None, scene=None, target_inc=None, target_azi=None, dls_design=3.,
    dls_max=5., generations=1, population=1000, survival_rate=10, decay=100.,
    inc_max=95, num_cpus=None, local_mode=False, new_species_rate=0.1
):
    """
    inc and azi in degrees
    """
    ray.init(
        local_mode=local_mode
    )

    sh = we.survey.SurveyHeader(
        azi_reference='grid'
    )
    survey_header = ray.put(sh)

    # initiate graph
    graph = Graph.remote()
    frame = Frames.remote()
    if bool(octree):
        octree_remote = ray.put(octree)
    else:
        octree_remote = octree
    cm = CollisionManager.remote(scene)

    # add nodes - start node id is 0 and target_id is 1
    graph.add_node.remote(
        pos=start_pos, inc=start_inc, azi=start_azi, md=start_md,
    )
    graph.add_node.remote(
        pos=target_pos, inc=target_inc, azi=target_azi, priority=np.inf
    )
    c_target = connect_graph_nodes(
        ray.get(graph.get_node.remote(0)),
        ray.get(graph.get_node.remote(1)),
        dls_design
    )
            # h = heuristic(
            #     G.graph.nodes[G.counter - 1]['pos'],
            #     target_pos
            # )
    heuristic = c_target.md_target - c_target.md1
    graph.set_attributes.remote({0: {'priority': heuristic}})
    # graph.get_priority.remote(0, set_value=heuristic)

    # G.graph.nodes[0]['priority'] = heuristic(
    #     G.graph.nodes[0]['pos'],
    #     G.graph.nodes[1]['pos']
    # )

    kwargs = dict(
        graph=graph,
        frame=frame,
        octree=octree_remote,
        cm=cm,
        survey_header=survey_header,
        target_inc=target_inc, target_azi=target_azi,
        dls_design=dls_design, dls_max=dls_max,
        population=population, generations=generations,
        survival_rate=survival_rate, decay=decay,
        inc_max=inc_max,
        new_species_rate=new_species_rate
    )

    if num_cpus is None:
        cpus = psutil.cpu_count() - 1
    else:
        cpus = num_cpus

    for g in range(generations):
        print(f"Starting generation: {g}")
        results = ray.get([_rrt_star_search.remote(
                    generation=g, **kwargs
                ) for _ in range(cpus)])

    # pbar = tqdm(total=(generations * population))
    # for g in range(generations):
    #     print(f"Starting generation: {g}")
    #     for cpu in range(8):
    #         _rrt_star_search.remote(
    #             generation=g, **kwargs
    #         )

    #     while True:
    #         time.sleep(1)
    #         num_nodes = ray.get(graph.number_of_nodes.remote())
    #         pbar.n = num_nodes - 1
    #         pbar.update()
    #         if num_nodes > (g + 1) * population:
    #             break


    # pbar.close()

    # F.graph = G.graph

    results = Object(**ray.get(frame.get_data.remote()))
    results.graph = Object(**ray.get(graph.get_graph_data.remote())).graph

    return results


@ray.remote
def _rrt_star_search(
    generation, graph, frame, survey_header=None, octree=None, cm=None,
    target_inc=None, target_azi=None, dls_design=3., dls_max=5.,
    generations=1, population=1000, survival_rate=10, decay=100.,
    inc_max=95, new_species_rate=0.1,
    **kwargs
):
    while True:
        num_nodes = ray.get(graph.get_number_of_active_nodes.remote())
        # num_nodes = ray.get(graph.number_of_nodes.remote())
        if num_nodes > (generation + 1) * population:
            break

        # best_well = ray.get(frame.get_best_well.remote())
        try:
            best_well = ray.get(graph.get_best_well.remote())
            # best_well = single_source_dijkstra(
            #             ray.get(graph.get_graph.remote()),
            #             0,
            #             1
            #         )
            if best_well == np.inf:
                best_well = ray.get(frame.get_best_well.remote())
        except nx.exception.NetworkXNoPath:
            best_well = ray.get(frame.get_best_well.remote())

        # update tqdm bar
        # pbar.n = num_nodes
        # pbar.update()

        # current = np.argmin([
        #     G.graph.nodes[n]['priority'] for n in G.graph.nodes
        # ])
        # current = min(
        #     G.graph._node.keys(),
        #     key=(lambda x: G.graph._node[x]['priority'])
        # )

        # try and force an occasional new branch
        if random.random() < new_species_rate:
            current = 0
        else:
            current = ray.get(graph.get_best_node.remote())
            # current = ray.get(graph.get_current.remote())
        # current = frontier.get()

        # generate a node
        n1 = ray.get(graph.get_node.remote(current))
        pos, vec, delta_md = get_random_position(
            pos=n1['pos'],
            vec=n1['vec'],
            dls=dls_design,
            step_min=10
        )

        # check that new pos is in unoccupied space
        # if it's not, add current back to queue after increasing priority
        if bool(octree):
            if octree.findNode(pos) is None or len(octree.findNode(pos).data) > 0:
                graph.decay_node.remote(current, decay=decay * 2)
                # graph.get_priority.remote(current, decay=decay * 2)
                # p = G.graph.nodes[current]['priority'] + decay * 2
                # G.graph.nodes[current]['priority'] = p
                # frontier.put(current[1], p)
                frame.add_frame.remote(
                    # ray.get(frame.get_number_of_frames.remote()),
                    generation,
                    current,
                    best_well
                )
                continue

        # add node to Graph and get index
        new = ray.get(graph.add_node.remote(
            pos=pos, vec=vec, md=n1['md'] + delta_md,
        ))

        # get connector nodes
        c = connect_graph_nodes(
            ray.get(graph.get_node.remote(current)),
            ray.get(graph.get_node.remote(new)),
            dls_design,
            force_min_curve=True  # otherwise it tries to do 'curve_hold_curve"
        )
        s = c.survey(survey_header=survey_header)

        # check rules
        if np.max(s.inc_deg) > inc_max:
            graph.deactivate_node.remote(new)
            # graph.remove_node.remote(new)
            continue

        # graph.update_node.remote(
        #     idx=new, pos=c.pos_target, vec=c.vec_target, md=c.md_target,
        # )

        # assert np.isclose(c.md_target, ray.get(graph.get_node.remote(new))['md']), "Not close"

        if abs(n1['md'] + delta_md - c.md_target) > 1:
            print("We could have a problem")

        graph.update_node.remote(
                n=new, pos=c.pos_target, vec=c.vec_target, md=c.md_target,
            )

        graph.add_edge.remote(
                current,
                new,
                # weight=c.md_target - c.md1,
                survey=s,
                weight=np.inf,
                # mesh=m
            )

        # s = ray.get(graph.get_survey.remote(new))

        m = get_mesh(current, new, graph, survey_header)

        if m is None:
            # graph.remove_node.remote(new)
            # graph.get_priority.remote(current, set_value=np.inf)
            graph.deactivate_node.remote(new)
            graph.set_attributes.remote({current: {'priority': np.inf}})
            continue

        # path = single_source_dijkstra(
        #     ray.get(graph.get_graph.remote()),
        #     0,
        #     new
        # )

        # make survey
        # s = c.survey()

        # make mesh
        # TODO add WellMesh object to edges to include survey data
        # m = we.mesh.WellMesh(s, method='circle').mesh

        # check for collision
        if cm and ray.get(cm.in_collision_single.remote(m)):
            # graph.get_priority.remote(current, decay=decay)
            graph.decay_node.remote(current, decay=decay)
            # p = G.graph.nodes[current]['priority'] + decay
            # G.graph.nodes[current]['priority'] = p
            # graph.remove_node.remote(new)
            graph.deactivate_node.remote(new)
            frame.add_frame.remote(
                # ray.get(frame.get_number_of_frames.remote()),
                generation,
                current,
                best_well
            )
            continue
        else:
            # graph.add_edge.remote(
            #     current,
            #     new,
            #     weight=c.md_target,
            #     mesh=m
            # )
            # graph.update_edge_attributes.remote(
            #     current, new, attributes={
            #         'mesh': m, 'weight': c.md_target - c.md1
            #     }
            # )
            graph.set_attributes.remote(
                attributes={
                    (current, new):
                    {
                        'mesh': m, 'weight': c.md_target - c.md1
                    }
                }, property='ep'
            )
            cost = ray.get(graph.get_node.remote(new))['md']
            # cost = single_source_dijkstra(
            #     ray.get(graph.get_graph.remote()),
            #     0,
            #     new
            # )
            c_target = connect_graph_nodes(
                ray.get(graph.get_node.remote(new)),
                ray.get(graph.get_node.remote(1)),
                dls_design
            )
            s_target = c_target.survey(survey_header=survey_header)

            graph.add_edge.remote(
                new,
                1,
                # weight=c_target.md_target - c_target.md1,
                weight=np.inf,
                survey=s_target
                # mesh=m
            )
            m_target = get_mesh(new, 1, graph, survey_header)
            # h = heuristic(
            #     G.graph.nodes[G.counter - 1]['pos'],
            #     target_pos
            # )
            if m_target is None:
                graph.remove_node.remote(new)
                continue

            h = c_target.md_target - c_target.md1
            # graph.get_priority.remote(new, set_value=cost + h)
            graph.set_attributes.remote({new: {'priority': cost + h}})

            # G.graph.nodes[G.counter - 1]['priority'] = cost[0] + h
            # p = G.graph.nodes[current]['priority'] + decay
            # G.graph.nodes[current]['priority'] = p
            # graph.get_priority.remote(current, decay=decay)
            graph.decay_node.remote(current, decay=decay)
            frame.add_mesh.remote(m.mesh)
            frame.add_edge.remote((current, new))
            frame.add_frame.remote(
                    # len(F.meshes),
                    generation,
                    current,
                    best_well
                )

            if (
                np.amax(s_target.dls) > dls_max
                or np.amax(s_target.inc_deg) > inc_max
            ):
                graph.remove_edge.remote(new, 1)
                continue

            # check for valid edge to target
            # graph.add_edge.remote(
            #     new,
            #     1,
            #     weight=c_target.md_target - c_target.md1,
            #     survey=c_target.survey()
            #     # mesh=m
            # )
            # m_target = get_mesh(new, 1, graph, survey_header)
            # m_target = we.mesh.WellMesh(s_target, method='circle').mesh
            # graph.update_node.remote(
            #     idx=new, pos=c.pos_target, vec=c.vec_target, md=c.md_target,
            # )
            if ray.get(cm.in_collision_single.remote(m_target)):
                graph.remove_edge.remote(new, 1)
                continue
            else:
                # graph.add_edge.remote(
                #     new,
                #     1,
                #     weight=c_target.md_target,
                #     mesh=m_target
                # )
                graph.update_edge_attributes.remote(
                    new, 1, attributes={
                        'weight': c_target.md_target - c_target.md1
                    }
                )
                graph.set_attributes.remote(
                    attributes={
                        (new, 1): {'weight': c_target.md_target - c_target.md1}
                    },
                    property='ep'
                )
                # best_well = single_source_dijkstra(
                #     ray.get(graph.get_graph.remote()),
                #     0,
                #     1
                # )
                # G.graph.nodes[current]['priority'] = p
                frame.add_mesh.remote(m_target.mesh)
                frame.add_edge.remote((new, 1))
                frame.add_frame.remote(
                    # len(F.meshes),
                    generation,
                    current,
                    best_well
                )
    return


def get_mesh(n1, n2, graph, survey_header):
    # sh = ray.get(survey_header.remote())
    # if n2 == 1:
    #     s1 = ray.get(graph.get_survey.remote(0, n1))
    #     s2 = ray.get(graph.get_survey.remote(n1, n2))
    #     s = np.hstack((
    #         s1, s2
    #     ))
    # else:
    #     s = ray.get(graph.get_survey.remote(0, n2))
    # s = ray.get(graph.get_survey.remote(n1, n2))
    s = get_survey(graph, n1, n2)
    start_nev = ray.get(graph.get_node.remote(0))['pos']
    survey = we.survey.Survey(
        md=s[0], inc=s[1], azi=s[2], deg=False, error_model='iscwsa_mwd_rev4',
        start_nev=start_nev, header=survey_header
    )
    idx = np.where(np.isclose(
        survey.md, ray.get(graph.get_node.remote(n1))['md']
    ))
    sliced_survey = we.survey.slice_survey(
        survey, start=int(idx[0][0]), stop=None
    )
    # if (
    #     len(idx[0]) > 1
    #     and idx[0][1] - idx[0][0] > 1
    #     or not np.allclose(
    #         ray.get(graph.get_node.remote(n1))['pos'], np.array(sliced_survey.start_nev)
    #     )
    # ):
    #     print(
    #         f"({n1}, {n2})\n"
    #         f"{ray.get(graph.get_node.remote(n1))['pos']}, "
    #         f"{np.array(sliced_survey.start_nev)}"
    #     )
    #     return None
    # else:
    #     try:
    #         sliced_survey = we.survey.slice_survey(
    #             survey, start=int(idx[0][0]), stop=None
    #         )
    #     except:
    #         md_temp = ray.get(graph.get_node.remote(n1))['md']
    #         print(f'Help! ({n1}, {n2}), {idx}')
    #         print(f'{md_temp}')
    #         print(f'{survey.md}')
    #     assert np.allclose(
    #         ray.get(graph.get_node.remote(n1))['pos'], np.array(sliced_survey.start_nev)
    #     ), print(
    #         f"Oh no!"
    #         f"\n({n1}, {n2}), {idx}"
    #         f"\n{ray.get(graph.get_node.remote(n1))['md']}"
    #         f"\n{ray.get(graph.get_node.remote(n1))['pos']}"
    #         f"\n{np.array(sliced_survey.start_nev)}"
    #         )
    mesh = we.mesh.WellMesh(sliced_survey)

    return mesh


def get_survey(graph, n1, n2):
    graph = ray.get(graph.get_graph.remote())
    for path in gt.topology.all_paths(graph, 0, n2):
        if all((n1 in path, n2 in path)):
            break
    # path = [
    #     p for p in nx.all_simple_paths(
    #         graph, 0, n2)
    #     if n1 in p
    #     and n2 in p
    # ]
    temp = []
    for (u, v) in zip(path[:-1], path[1:]):
        # data = graph.get_edge_data(u, v)
        # temp.append([data['md'], data['inc'], data['azi']])
        data = {p: graph.ep[p][graph.edge(u, v)] for p in ['md', 'inc', 'azi']}
        temp.append([data['md'], data['inc'], data['azi']])

    return np.hstack(temp)


def a_star_search(octree, start_pos, start_inc, start_azi, target_pos, dls_max=5.):
    """
    inc and azi in degrees
    """
    # find the start and target nodes
    start_node = octree.findNode(start_pos)
    target_node = octree.findNode(target_pos)

    # initiate graph
    G = nx.Graph()
    G.add_node(start_node.idx)
    G.nodes[start_node.idx]['inc'] = start_inc
    G.nodes[start_node.idx]['azi'] = start_azi
    G.nodes[start_node.idx]['md'] = 0.

    # initiate queue
    frontier = PriorityQueue()
    frontier.put(start_node.idx, 0)

    # initiate cost dictionaries
    came_from = {}
    cost_so_far = {}

    came_from[start_node.idx] = None
    cost_so_far[start_node.idx] = 0

    while not frontier.empty():
        current = frontier.get()
        if current == target_node.idx:
            break
        current_node = octree.get_node_from_idx(current)

        # get neighbors
        neighbors = octree.findNeighbors(
            node=current_node,
            inc=G.nodes[current]['inc'],
            azi=G.nodes[current]['azi']
        )

        for n in neighbors:
            # create connector nodes
            n1, n2 = get_connector_nodes(
                current_node, n, G.nodes[current]
            )
            data = get_cost(n1, n2, dls_max)
            if not data:
                continue
            else:
                n_cost, n_inc, n_azi = data

            new_cost = (
                cost_so_far[current]
                + n_cost
            )

            if n.idx in cost_so_far:
                if new_cost > cost_so_far[n.idx]:
                    continue
            else:
                cost_so_far[n.idx] = new_cost
                priority = new_cost + heuristic(
                    octree.get_node_from_idx(current).position,
                    target_node.position
                )
                frontier.put(n.idx, priority)
                came_from[n.idx] = current
                G.add_edge(current, n.idx, weight=n_cost)
                G.nodes[n.idx]['md'] = n_cost
                G.nodes[n.idx]['inc'] = n_inc
                G.nodes[n.idx]['azi'] = n_azi

    sp = single_source_dijkstra(G, start_node.idx, target_node.idx)

    idx, pos = get_path(
        octree, sp
    )

    bounds = get_bounds(octree, idx)

    return came_from, cost_so_far, idx, pos, bounds


def get_random_position(
    pos, vec, dls=3.0, probability=0.5, nev=True, unit='meters', step_min=30.
):
    """
    Get a random position based on the current position and vector and
    bound by a function of the Dog Leg Severity. The result is discrete in
    being either straight or along the path the DLS radius, the probability
    of which is defined by probability.

    Parameters
    ----------
        pos: (1, 3) array of floats
            The current position.
        vec: (1, 3) array of floats
            The current vector.
        dls: float
            The desired Dog Leg Severity in degrees per 30 meters or 100 feet.
        probability: float between 0 and 1.
            The probability of the returned position being straight ahead
            versus being along a circular path with the radius defined by dls.
        nev: bool (default: True)
            Indicates whether the pos and vec parameters are provided in the
            [North, East, Vertical] or [x, y, z] coordinate system.
        unit: string (default: 'meters')
            The unit, either 'meters' or 'feet'.
        step_min: float (default: 30)
            The minimum allowed delta_md

    Returns
    -------
        pos_new: (1, 3) array of floats
            A new position.
        vec_new: (1, 3) array of floats
            A new vector
    """
    assert unit in ['meters', 'feet'], "Unit must be 'meters' or 'feet'"
    pos_nev, pos_xyz = utils.process_coords(pos, nev)
    vec_nev, vec_xyz = utils.process_coords(vec, nev)

    if unit == 'meters':
        coeff = 30
    else:
        coeff = 100

    radius = (360 * coeff) / (dls * 2 * np.pi)
    # delta_md = random.random() * radius
    factor_min = step_min / (math.pi * radius * 0.5)
    factor = random.random()
    factor = factor_min if factor < factor_min else factor
    dogleg = factor * math.pi / 2
    delta_md = factor * math.pi * radius * 0.5
    # if delta_md < step_min:
    #     delta_md = step_min
    action = 0 if random.random() < probability else 1

    if action:
        # do some directional drilling
        # dogleg = math.atan(delta_md / radius)
        pos_temp = np.array([
            math.cos(dogleg),
            0.,
            math.sin(dogleg)
        ]) * radius
        pos_temp[0] = radius - pos_temp[0]

        vec_temp = np.array([
            math.sin(dogleg),
            0.,
            math.cos(dogleg)
        ])

        # spin that roulette wheel to see which direction we're heading
        toolface = random.random() * 2 * np.pi
        inc, azi = utils.get_angles(vec_nev, nev=True).reshape(2)
        angles = [
            toolface,
            inc,
            azi
        ]

        # Use the code from those clever chaps over at scipy, I'm sure they're
        # better at writing this code than I am.
        r = R.from_euler('zyz', angles, degrees=False)

        pos_new, vec_new = r.apply(np.vstack((pos_temp, vec_temp)))
        pos_new += pos_nev

    else:
        # hold section
        vec_new = vec_nev
        pos_new = pos_nev + (vec_nev * delta_md)

    if not nev:
        pos_new, vec_new = utils.get_xyz(
            np.vstack((pos_new, vec_new))
        )

    return (pos_new, vec_new, delta_md)


# def get_path(octree, start_node, target_node, came_from):
#     idx = [target_node.idx]
#     pos = [target_node.position]

#     while True:
#         if came_from[idx[-1]] == start_node.idx:
#             return (idx, pos)
#         idx.append(came_from[idx[-1]])
#         pos.append(
#             octree.get_node_from_idx(came_from[idx[-1]]).position
#         )


def get_path(octree, path):
    pos = np.array([
        octree.get_node_from_idx(idx).position
        for idx in path[1]
    ])

    return path[1], pos


def get_bounds(octree, idx):
    bounds = []
    for i in idx:
        n = octree.get_node_from_idx(i)
        bounds.extend(
            [(p - n.size / 2, p + n.size / 2) for p in n.position]
        )
    return bounds


def random_well(
    pos=[0., 0., 0.], vec=[0., 0., 1.], dls_design=3.0, steps=10
):
    poss = [pos]
    vecs = [vec]
    for i in range(steps):
        pos_new, vec_new = get_random_position(
            poss[-1],
            vecs[-1],
            probability=0.3,
        )
        poss.append(pos_new)
        vecs.append(vec_new)

    return poss


if __name__ == "__main__":
    import ray
    import os

    os.environ['DISPLAY'] = ':1'

    ray.init()

    @ray.remote
    def worker():
        p = random_well()
        s = connector.connect_points(p)
        m = we.mesh.WellMesh(s, method='circle').mesh

        return m

    meshes = [worker.remote() for _ in range(100)]

    we.visual.plot(data=ray.get(meshes))

    print("Done")
