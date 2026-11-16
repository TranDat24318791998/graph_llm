"""Type aliases for the road graph."""

import networkx as nx

# The road network as OSMnx builds it: a directed multigraph whose nodes are OSM node ids.
type RoadGraph = nx.MultiDiGraph[int]
