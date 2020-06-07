def Topk(List, k, reverse=False):
    """
    return the top k item in List and their indexes. If reverse, return the least k items
    """
    List = list(List)
    if len(List) < k:
        raise ValueError("The lenth of list is smaller than {}".format(k))
    topk_list = []
    index_list = []
    # if we want the least k items, replace by the largest value. if we want the biggest k items, replace by the smallest values.
    if reverse:
        replace_value = max(List)
    else:
        replace_value = min(List)
    for _ in range(k):
        # if we want the least k items, we want the smallest value. else if we want the biggest k items, we want the biggest value.
        if reverse:
            find_value = min(List)
        else:
            find_value = max(List)
        topk_list.append(round(find_value, 4))
        # find the index of the first item that equal to find_value
        find_index = List.index(find_value)
        while (find_index in index_list):
            find_index = List[find_index + 1, :].index(find_value) + find_index + 1
        index_list.append(int(find_index))
        List[find_index] = replace_value
    return topk_list, index_list


