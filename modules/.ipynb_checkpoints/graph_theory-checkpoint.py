import graph_tool as gt
from graph_tool.topology import subgraph_isomorphism


def extract_circular_subgraph_vertices_and_edges(mol, center_atom_index, radius):
    
    V = {center_atom_index}
    E = set({})
    
    for r in range(radius):
        for current_atom_index in V:
            
            current_neighbours = {atom.GetIdx() for atom in mol.GetAtomWithIdx(current_atom_index).GetNeighbors()}
            V = V.union(current_neighbours)
            E = E.union({tuple(sorted([current_atom_index, neighbour_index])) for neighbour_index in current_neighbours})

    return V, E



def get_bond_invs(mol, bond_atom_ids, use_bond_invs, use_chirality):
    
    bond = mol.GetBondBetweenAtoms(bond_atom_ids[0], bond_atom_ids[1])
    
    if use_bond_invs == False:
        
        return 1
    
    else:
        
        if use_chirality == False:
            
            return abs(hash((str(bond.GetBondType()),)) % 10000000)
        
        else:
            
            if str(bond.GetBondType()) == "DOUBLE":
            
                return abs(hash((str(bond.GetBondType()), str(bond.GetStereo()))) % 10000000)
            
            else:
                
                return abs(hash((str(bond.GetBondType()),)) % 10000000)



def extract_labelled_circular_subgraph_object(mol, center_atom_index, radius, ecfp_settings):
    
    # create vertices and edges
    (V, E) = extract_circular_subgraph_vertices_and_edges(mol, center_atom_index, radius)

    # create vertex label dict
    vertices_to_labels = {}
    invs = ecfp_settings["mol_to_invs_function"](mol)
    
    for v in V:
        vertices_to_labels[v] = invs[v]

    # create edge label dict
    edges_to_labels = {}
    
    for (v, w) in E:
        edges_to_labels[(v, w)] = get_bond_invs(mol = mol, bond_atom_ids = (v, w), use_bond_invs = ecfp_settings["use_bond_invs"], use_chirality = ecfp_settings["use_chirality"])
    
    # standardise graph data via isomorphism so that the n vertices are mapped to {0,...,n-1}
    isom = dict(list(zip(list(V), list(range(len(V))))))
    V = set({isom[v] for v in V})
    E = set({(isom[v], isom[w]) for (v, w) in E})
    vertices_to_labels = {isom[v]: vertices_to_labels[v] for v in vertices_to_labels.keys()}
    edges_to_labels = {(isom[v], isom[w]) : edges_to_labels[(v, w)] for (v, w) in edges_to_labels.keys()}
    
    # create topological graph object
    G = gt.Graph(directed = False)
    
    G.add_vertex(n = len(V))
    
    for (v,w) in E:
        G.add_edge(v, w)

    # create vertex property map
    vertex_labels = G.new_vertex_property("long")
    
    for vertex in G.vertices():
        vertex_labels[vertex] = vertices_to_labels[int(vertex)]
    
    # create edge property map
    edge_labels = G.new_edge_property("long")
    
    for edge in G.edges():
        edge_labels[edge] = edges_to_labels[tuple(edge)]
    
    return (G, vertex_labels, edge_labels)



def check_if_strict_labelled_subgraph(G_sub_with_labels, G_with_labels):
    
    (G_sub, G_sub_vertex_labels, G_sub_edge_labels) = G_sub_with_labels
    (G, G_vertex_labels, G_edge_labels) = G_with_labels
    
    if G_sub.num_vertices() >= G.num_vertices():
        
        return False
    
    else:
    
        list_of_vertex_maps = subgraph_isomorphism(G_sub, 
                                                   G, 
                                                   max_n = 1, 
                                                   vertex_label = (G_sub_vertex_labels, G_vertex_labels), 
                                                   edge_label = (G_sub_edge_labels, G_edge_labels), 
                                                   induced = False)

        return True if len(list_of_vertex_maps) > 0 else False