import warnings


def determine_list_datatype(l: list, string_type=str):
    l_filtered = [el for el in l if el is not None]
    if all(isinstance(el, type(l_filtered[0])) for el in l_filtered):
        dt = type(l_filtered[0]) if not isinstance(l_filtered[0], str) else string_type
    else:
        warnings.warn("While determining the type of elements in the list: not all elements are of the same type. "
                      "Default type in this case is string for all elements.", RuntimeWarning)
        dt = string_type
    return dt