
def center_refine(center_coordinate, radius):
    '''
    In order to avoid the crop out of boundary
    :param center_coordinate:
    :param radius:
    :return:
    '''
    if (center_coordinate - radius) < 1:
        center_coordinate += (radius - center_coordinate + 5)
    elif (center_coordinate + radius) > 255:  # maybe need modification for each size of img
        center_coordinate -= (center_coordinate + radius - 255 + 5)
    return center_coordinate