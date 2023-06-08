def node_sum(dend, val_dict):
    """ given item per leaf in tree, sums leaf values and adds to dictionary

    Args:
        dend (np.array): (2, num_leaf) dendrogram arrays (equiv to
            sklearn.cluster.Ward.children_)
        val_dict (dict): keys are leaf indices (0 to num_leaf), values
            are items to be summed

    Returns:
        val_dict (dict): same structure as input, but now includes all
            nodes, not just leafs
    """
    num_leaf = len(val_dict)
    for node_idx, (c0, c1) in enumerate(dend.T):
        node_idx += num_leaf
        new_val = val_dict[c0] + val_dict[c1]
        val_dict[node_idx] = new_val

    return val_dict
