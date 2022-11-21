def ZeroBubble(patch_center, attenm_img_np, radius):
    '''

    :param patch_center:
    :param attenm_img_np:
    :param radius:
    :return:
    '''
    # check limit
    x_leftlim = patch_center[0] - radius
    x_rightlim = patch_center[0] + radius
    y_leftlim = patch_center[1] - radius
    y_rightlim = patch_center[1] + radius
    z_leftlim = patch_center[2] - radius
    z_rightlim = patch_center[2] + radius
    if x_leftlim <= 0:
        x_leftlim = 0
    if x_rightlim >= attenm_img_np.shape[0]:
        x_rightlim = attenm_img_np.shape[0]
    if y_leftlim <= 0:
        y_leftlim = 0
    if y_rightlim >= attenm_img_np.shape[1]:
        y_rightlim = attenm_img_np.shape[1]
    if z_leftlim <= 0:
        z_leftlim = 0
    if z_rightlim >= attenm_img_np.shape[2]:
        z_rightlim = attenm_img_np.shape[2]

    for xcoord in range(x_leftlim, x_rightlim + 1):
        for ycoord in range(y_leftlim, y_rightlim + 1):
            for zcoord in range(z_leftlim, z_rightlim + 1):
                attenm_img_np[xcoord, ycoord, zcoord] = 0

    return attenm_img_np