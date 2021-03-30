"""
Most part are adopted from original louvain package from pypi.
See https://pypi.org/project/python-louvain/ for original package.


The Louvain algorithm is an algorithm for finding communities in a un-directed network.

Modularity :       sum of each communities' (sum_in_c/m - (total_c/2m)**2)
m :                total edges in network.
sum_in_c :         total edges within community c. one-side, means observed edges.
sum_in_c/m :       observed degrees' percentage.i.e. .(2*sum_in_c / 2*m) equals to (sum_in_c/m).
(total_c/2m)**2 :  expected degrees' percentage

The Louvain algorithm starts from a singleton partition in which each node is in its own community .
(a). The algorithm moves individual nodes from one community to another to find a partition .
(b). Based on this partition, an aggregate network is created .
(c). The algorithm then moves individual nodes in the aggregate network.
(d). These steps are repeated until the quality cannot be increased further.

Reference:
    http://iopscience.iop.org/article/10.1088/1742-5468/2008/10/P10008/meta
    http://bora.uib.no/bitstream/handle/1956/16057/Thesis_Herman_Lund.pdf
    https://pypi.org/project/python-louvain/

Recommendation:
    Leiden algorihtm ,which is a extension of Louvain algorihtm.
    https://github.com/vtraag/leidenalg
    Traag, V.A., Waltman. L., Van Eck, N.-J. (2018). From Louvain to Leiden: guaranteeing well-connected communities. arXiv:1810.08473
"""
from __future__ import division

from collections import defaultdict
from copy import deepcopy

import numpy as np

MIN_DELTA = 0.0000001


def count(func):
    def c(self, *args):
        import time
        st = time.time()
        res = func(self, *args)
        print("cost: {}".format(time.time() - st))
        return res

    return c


class Louvain(object):
    """
    Louvain algorithm.
    """

    def __init__(self, graph, partition=None, res=1., random_state=None):
        """
        Find best community partition using Louvain algorithm.

        :param graph: non-direction networkx.Graph.
        :param partition: dict of each node's community tag.
        :param res: user-specific resolution,infulence final partition.
        :param random_state: np.random.RandomState/int/np.integer/np.random.mtrand._rand.
        """
        self.graph = graph
        self.resolution = res
        self.random_state = self._check(random_state, graph)
        self.partition_supplied = bool(partition)
        self._parts = []
        self._parts.append(Partition(graph, partition))

    @staticmethod
    def _check(random_state, graph):
        """
        Check graph and randomstate.

        :param random_state:  np.random.RandomState/int/np.integer/np.random.mtrand._rand
        :param graph: none-direction nwtworkx.Graph
        :return: None.
        """
        if graph.is_directed() or graph.number_of_edges() == 0:
            raise TypeError("Use only non-directed graph and there should be at least one links within nodes")

        if random_state is None:
            return np.random.mtrand._rand
        if isinstance(random_state, (int, np.integer)):
            return np.random.RandomState(random_state)
        if isinstance(random_state, np.random.RandomState):
            return random_state

        raise ValueError("%r cannot be used to seed a numpy.random.RandomState instance".format(random_state))

    @count
    def split(self):
        """
        Split graph using Louvain algorithm.

        :return: best partition dict (at maximum level), a dict of each node's community tag.equal to Louvain.get_partition()
        """
        cur_graph = self.graph.copy()
        cur_part = self._parts[0].copy()
        cur_mod = cur_part.modularity

        while 1:
            # first step of Louvain algorithm.
            new_part = self._find_best_partition(self.random_state, cur_part, self.resolution)
            new_part.sort()
            self._parts.append(new_part)
            new_mod = new_part.modularity

            if new_mod - cur_mod < MIN_DELTA:
                break

            cur_mod = new_mod
            # second step of Louvain algorithm.
            new_graph = self.combine_nodes(cur_graph, new_part.partition)
            cur_graph = new_graph
            cur_part = Partition(new_graph)
        # last partion have not been optimised,just be combined.
        del self._parts[-1]

        return self.get_partition()

    @staticmethod
    def _find_best_partition(random_state, part, resolution=1.):
        """
        First step (Core) of Louvain algorithm.\n
        Iteration through all nodes till a local maximum of modularity is attained. i.e.\n
        When no individual move can improve the modularity(delta should be positive and higher than a threshhold)\n

        :param random_state: np.random.RandomState,used for generating random permutation of all nodes.
        :param part: Partition object.
        :return: Partition object of new partition which cause the maximum mudularity increment.
        """
        modified = True
        new_mod = part.modularity
        while modified:
            cur_mod = new_mod
            modified = False
            for node in random_state.permutation(list(part.node_com.keys())):
                maximum_delta = 0.
                com1 = part[node]
                best_com = com1
                ki_d_2m = part.node_tot[node] / (2. * part.total_weight)

                neignbor_communities = part.neigbor_communities(node)
                remove_decrease = (part.com_tot[com1] - part.node_tot[node]) * ki_d_2m \
                                  - resolution * neignbor_communities[com1]
                # Remove must be executed after calculating decrease,
                # whereas it could be executed after or before calculating delta.
                # It would be the best to put this in the middle of this two calculation.
                # As a consequence,delta will be zero when remove a node from a community and put it back.)
                for com2, k_in2 in random_state.permutation(list(neignbor_communities.items())):
                    # delta equals to decrease + increase.
                    if com2 == com1:
                        continue
                    delta = remove_decrease + (resolution * k_in2 - (part.com_tot[com2]) * ki_d_2m)
                    if delta > maximum_delta:
                        maximum_delta = delta
                        best_com = com2

                # Insert can only be executed after waiting for all calculation done.
                part.move(node, best_com, neignbor_communities[com1], neignbor_communities[best_com])
                if best_com != com1:
                    modified = True
            new_mod = part.modularity
            if new_mod - cur_mod < MIN_DELTA:
                break

        return part

    @staticmethod
    def combine_nodes(graph, partition):
        """
        @staticmethod
        Combine nodes share the same community tag into a single node.\n
        Return a new graph constructed from single nodes.\n

        :param graph: networkx.Graph.
        :param partition: dict of each node's community tag.
        :return: networkx.Graph.
        """
        if not isinstance(partition, dict):
            raise TypeError("partition support dict or Partition object")
        new_graph = nx.Graph()
        new_graph.add_nodes_from(set(partition.values()))

        for node1, node2, datas in graph.edges(data=True):
            edge_weight = datas.get('weight', 1)
            com1, com2 = partition[node1], partition[node2]
            remain = new_graph.get_edge_data(com1, com2, {'weight': 0}).get('weight', 1)
            new_graph.add_edge(com1, com2, **{'weight': remain + edge_weight})

        return new_graph

    @property
    def partitions(self):
        """
        List of each level's Partition object.\n
        Be cautious that number of nodes becomes smaller with the increase of level caused by 'combine' function.\n
        There is a equal relation between tags of current level's partition and nodes of the next level.\n
        Use Louvain.get_partition(level=None) instead under normal circumstances.\n

        :return: a list of each level's Partition object.
        """
        return self._parts.copy()

    def get_partition(self, level=None):
        """
        Fetch partition at each level.level = <0 - len(self.partitions)-1>

        :param level: level 0 represent the initial partition.level len(self.partitions)-1 represent the final partition.
        :return: dict of each node's community tag in the state of specified level.
        """
        max_level = len(self._parts) - 1
        if level is None:
            level = max_level
        if not (0 <= level <= max_level):
            raise ValueError("level must between <{}-{}>".format(0, max_level))
        partition_copy = self._parts[1].partition.copy()

        for node in self._parts[1].partition.keys():
            for _level in range(2, level + 1):
                com = partition_copy[node]
                partition_copy[node] = self._parts[_level].partition[com]

        return partition_copy

    @staticmethod
    def get_modularity(graph, partition):
        """
        @staticmethod\n
        Calculate modularity of this partition.\n
        This could be time-consuming.\n
        Use Partition.modularity instead if there is a Partition object.\n

        :param graph: networkx.Graph
        :param partition: dict of each node's community tag.
        :return: modularity of this partition.
        """
        part = Partition(graph=graph, partition=partition)

        return part.modularity


class Partition(object):
    """
    Partition object for containing each partition's information which will be used in Louvain algorithm.\n
    Contains four properties:\n
    type: default_dict(int).\n
    0: total_weight(float/int)  total weights of edges within this graph, i.e. (m).\n
    1: node_com a partition dict of each node's community tag.\n
    2: node_in  weight of edge from this node to this node,i.e. self-loop weight.\n
    3: node_tot sum weights of edges from all nodes in graph to this node.i.e total degree(one-side).\n
    4: com_in   sum weights of edges within this community.\n
    5: com_tot  sum weights of edges from all nodes in graph to nodes in this community.i.e. total degree(one-side).\n
    """
    __slots__ = ("level", "total_weight", "_graph", "_node_com", "_node_edges_in", "_node_degree_tot",
                 "_com_edges_in", "_com_degree_tot")

    def __init__(self, graph=None, partition=None, level=0):
        """
        :param graph: networkx.Graph.
        :param level: int, partition level.0 is the minimum level means each single node is a single community.
        :param partition: dict of each node's community tag. user supplied partion information.
        """
        self.level = level
        self.total_weight = 0
        self._graph = graph
        self._node_com = {}
        self._node_degree_tot = defaultdict(int)
        self._node_edges_in = defaultdict(int)
        self._com_degree_tot = defaultdict(int)
        self._com_edges_in = defaultdict(int)
        self._init(graph, partition)

    def _init(self, graph, partition):
        """
        Calculation of partition's information.

        :param graph: partition's graph.
        :param partition: dict of each node's community tag.  user supplied partion information.
        :return: None.
        """
        if graph is None:
            return

        if partition is not None and any((not isinstance(value, int) for value in partition.values())):
            raise ValueError("Community tags must be intergers")

        un_partitioned = list(graph.nodes())
        start_com = 0
        # Use partition user-provided information to set some node's community.
        if partition is not None:
            for node, com in partition:
                try:
                    un_partitioned.remove(node)
                except ValueError:
                    raise ValueError("Node in partion not exist in graph")

                self._node_com = graph.degree(node, weight='weight')
                degree_tot = graph.degree(node, weight='weight')
                self._node_degree_tot[node] = degree_tot
                self._node_edges_in[node] = graph.get_edge_data(node, node, default={'weight': 0}).get("weight", 1)
                self._com_degree_tot += self._com_degree_tot[com] + degree_tot
                # Calculate each community's sum of inside links.
                inc = 0.
                for neighbor, datas in graph[node].items():
                    neighbor_com = partition.get(neighbor, None)
                    if neighbor_com is not None and neighbor_com == com:
                        edge_weight = datas.get('weight', 1)
                        if neighbor == node:
                            # Meet one specific link(self to self) only once.
                            inc += float(edge_weight)
                        else:
                            # Meet one specific link(one to another) two times.
                            inc += float(edge_weight) / 2.
                self._com_edges_in += inc
            start_com = max(partition.values()) + 1

        # Set each remained(not supplied information of community from partition) node to an isolate community.
        com = start_com
        for node in un_partitioned:
            self._node_com[node] = com
            degree_tot = graph.degree(node, weight='weight')
            degree_in = graph.get_edge_data(node, node, default={'weight': 0}).get("weight", 0)
            self._node_degree_tot[node] = degree_tot
            self._node_edges_in[node] = degree_in
            self._com_degree_tot[com] = degree_tot
            self._com_edges_in[com] = degree_in
            com += 1

        self.total_weight = graph.size(weight='weight')

    def move(self, node, com2, ki_in1=None, ki_in2=None):
        """
        Move a node into a different community and refresh partition's attribute.

        :param node: source node.
        :param com2: objective community which node be moved in.
        :param ki_in1: sum of the weights of links from node to nodes in this node's original community.
        :param ki_in2: sum of the weights of links from node to noded in objective community(com2).
        :return: return True if move successfully; return False when com1 == com2 .
        """
        if self.node_com[node] != com2:
            self.remove(node, ki_in1)
            self.insert(node, com2, ki_in2)

    def remove(self, node, ki_in1=None):
        """
        Remove a node from it's original community and refresh partition's attribute.

        :param node: node which should be removed from it's original community.
        :param ki_in1: sum of the weights of links from node to nodes in this node's original community.
                       links not include self-loop.
                       see Partition.neigbor_communities() for detailed reason.
        :return: None.
        """
        com1 = self._node_com[node]
        # if information of sums of weights not supplied,fetch information from self.neighbor_communities().
        if ki_in1 is None:
            ki_in1 = self.neigbor_communities(node)[com1]
        self._com_degree_tot[com1] = self.com_tot[com1] - self.node_tot[node]
        # com_in contains self-loop
        self._com_edges_in[com1] = self.com_in[com1] - ki_in1

    def insert(self, node, com2, ki_in2=None):
        """
        Insert a new node into a community and refresh partition's attribute.

        :param node: node which should be moved into community2.
        :param com2: objective community which node should be moved in.
        :param ki_in2: sum of the weights of links from node to nodes in objective community(com2).
                       links not include self-loop.
                       see Partition.neigbor_communities() for detailed reason.
        :return: None.
        """
        # if information of sums of weights not supplied,fetch information from self.neighbor_communities().
        if ki_in2 is None:
            ki_in2 = self.neigbor_communities(node)[com2]
        self._com_degree_tot[com2] = self.com_tot[com2] + self.node_tot[node]
        # com_in contains self-loop
        self._com_edges_in[com2] = self.com_in[com2] + ki_in2
        self._node_com[node] = com2

    def neigbor_communities(self, node):
        """
        Calculate ki_in of node to each neighboring communities.\n
        Pay attention to the definition of ki_in and sum_in:\n
        ki_in:
            When we calculate neigbor_communitys.There has a disequilibrium between original community and other community.\n
            Which is because self-loop will be encountered when scanning one node's neighbors in it's original community.\n
            While there is no chance of adding self-loop when calculating neighbors in other community.\n
            Thus we don't count self-loop in this step and self-loop add (-self-loop) will be zero in calculation of delta.\n
            Whereas self-loop should not be ignored in the step of updating information in remove and insert.\n
        sum_in:
            both com_in and node_in contain self-loop.\n

        :param node: source node.
        :return: dict of sum of the weights from node to nodes in each neighboring community(include self's community).
        """
        weights = defaultdict(int)
        for neighbor, datas in self._graph[node].items():
            # ki_in not contains self loop.
            neighbor_com = self._node_com[neighbor]
            edge_weight = datas.get('weight', 1)
            weights[neighbor_com] = weights[neighbor_com] + edge_weight

        for com, value in weights.items():
            if com != self.node_com[node]:
                value += self.node_in[node]

        return weights

    def copy(self):
        part_copy = deepcopy(self)
        part_copy.level += 1
        return part_copy

    def sort(self):
        """
        This method will call Partition.sort_tag to renumber self._node_com.See Partition.sort_tag for more information.

        :return: None.
        """
        self._node_com = self.sort_tag(self._node_com)

    @staticmethod
    def sort_tag(old_node_com):
        """
        @staticmethod\n
        Renumber all communities' tags to consecutive intergers. i.e. 0-N.\n

        :param old_node_com: a dict of each node's community tag.
        :return: a dict of renumbered node_com.
        """
        new_node_com = old_node_com.copy()
        new_tags = {}
        start_com = 0
        for node, com in old_node_com.items():
            new_com = new_tags.get(com, None)
            if new_com is None:
                new_tags[com] = start_com
                new_com = start_com
                start_com += 1
            new_node_com[node] = new_com

        return new_node_com

    @property
    def node_com(self):
        return self._node_com

    @property
    def com_in(self):
        return self._com_edges_in

    @property
    def com_tot(self):
        return self._com_degree_tot

    @property
    def node_in(self):
        return self._node_edges_in

    @property
    def node_tot(self):
        return self._node_degree_tot

    @property
    def modularity(self):
        """
        Calculate modularity of this partition.

        :return: modularity.
        """
        total_weight = self.total_weight
        modu = 0.
        for com, sum_in in self.com_in.items():
            # In original Louvain paper,sum_in means total degrees(two-side).here it represents total edges().
            modu += sum_in / total_weight - (self.com_tot[com] / (2. * total_weight)) ** 2.
        print("modularity", modu)
        return modu

    @property
    def partition(self):
        return self.node_com

    def __getitem__(self, node):
        """
        :param node: source node.
        :return: community tag of this node.
        """
        if node not in self.node_com.keys():
            raise ValueError("node not exists in this partition")
        return self.node_com[node]

    def __repr__(self):
        return repr(self.node_com)


if __name__ == "__main__":
    import networkx as nx
    import time
    import matplotlib.pyplot as plt

    colormap = plt.cm.gist_ncar
    # generate graph randomly.
    G = nx.erdos_renyi_graph(100, 0.05, seed=2)
    louvain = Louvain(G, res=1.0, random_state=1)
    part = louvain.split()
    partitions = louvain.partitions
    for level, _partition in enumerate(partitions):
        print("level:{} ".format(level), _partition)
        print(len(_partition.partition.values()))
    print("final partition: ", part)

    # plot graph.
    size = len(set(part.values()))
    colors = [colormap(i) for i in np.linspace(0, 0.9, size)]
    print("final number of communities: ", size)
    pos = nx.spring_layout(G)
    for count, com in enumerate(set(part.values())):
        list_nodes = [nodes for nodes in part.keys() if part[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=20, node_color=colors[count])
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
