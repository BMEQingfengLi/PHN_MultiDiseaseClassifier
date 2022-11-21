from Utils.CenterRefine import center_refine
import numpy as np

def croppatchacrosscenter(center_coordinate_np, origin_img_np, patch_radius):
    '''

    :param center_coordinate_np:
    :param origin_img_np:
    :param patch_radius:
    :return:
    '''

    patch_center_x = center_refine(center_coordinate_np[0], patch_radius)
    patch_center_y = center_refine(center_coordinate_np[1], patch_radius)
    patch_center_z = center_refine(center_coordinate_np[2], patch_radius)
    patch_np = origin_img_np[(patch_center_x - patch_radius):(patch_center_x + patch_radius),
                             (patch_center_y - patch_radius):(patch_center_y + patch_radius),
                             (patch_center_z - patch_radius):(patch_center_z + patch_radius)]

    return patch_np


def croppatchacrosscenter_probmap(center_coordinate_np, origin_img_np, patch_radius):
    '''

    :param center_coordinate_np:
    :param origin_img_np:
    :param patch_radius:
    :return:
    '''

    patch_center_x = center_refine(center_coordinate_np[0], patch_radius)
    patch_center_y = center_refine(center_coordinate_np[1], patch_radius)
    patch_center_z = center_refine(center_coordinate_np[2], patch_radius)
    patch_np = origin_img_np[(patch_center_x - patch_radius):(patch_center_x + patch_radius),
                             (patch_center_y - patch_radius):(patch_center_y + patch_radius),
                             (patch_center_z - patch_radius):(patch_center_z + patch_radius)]

    return patch_np, [patch_center_x, patch_center_y, patch_center_z]


def croppatchacrosscenter_inGenerateFeature(center_coordinate_np, origin_img_np, patch_radius):
    '''

    :param center_coordinate_np:
    :param origin_img_np:
    :param patch_radius:
    :return:
    '''

    patch_center_x = center_refine(center_coordinate_np[0], patch_radius)
    patch_center_y = center_refine(center_coordinate_np[1], patch_radius)
    patch_center_z = center_refine(center_coordinate_np[2], patch_radius)
    patch_np = origin_img_np[(patch_center_x - patch_radius):(patch_center_x + patch_radius),
               (patch_center_y - patch_radius):(patch_center_y + patch_radius),
               (patch_center_z - patch_radius):(patch_center_z + patch_radius)]

    center_coordinate_np = np.array([patch_center_x, patch_center_y, patch_center_z])

    return patch_np, center_coordinate_np