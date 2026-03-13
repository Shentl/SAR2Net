import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Callable

### External Imports ###
import torch as tc

### Internal Imports ###
from dhr_utils import utils as u

from dhr_initial_alignment import exhaustive_rigid_search as ers
from dhr_initial_alignment import sift_ransac as sr
from dhr_initial_alignment import sift_superglue as ssg
from dhr_initial_alignment import superpoint_ransac as spr
from dhr_initial_alignment import superpoint_superglue as spsg
from dhr_initial_alignment import feature_combination as fc
from dhr_initial_alignment import io_affine as ioa
from dhr_initial_alignment import multi_feature as mf
########################

### Algorithms ###

# TODO - description of all the methods

def identity_initial_alignment(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return u.create_identity_transform(source)

def exhaustive_rigid_search(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return ers.exhaustive_rigid_search(source, target, params)

def sift_ransac(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return sr.sift_ransac(source, target, params)

def sift_superglue(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return ssg.sift_superglue(source, target, params)

def superpoint_ransac(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return spr.superpoint_ransac(source, target, params)

def superpoint_superglue(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return spsg.superpoint_superglue(source, target, params)

def feature_combination(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return fc.feature_based_combination(source, target, params)

def rotated_feature_combination(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return fc.rotated_feature_based_combination(source, target, params)

def multi_feature(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return mf.multi_feature(source, target, params)

def instance_optimization_affine_registration(
    source : tc.Tensor,
    target : tc.Tensor,
    initial_displacement_field : Union[tc.Tensor, None],
    params : dict) -> tc.Tensor:
    return ioa.instance_optimization_affine_registration(source, target, initial_displacement_field, params)

def get_function(function_name : str) -> Callable:
    return getattr(current_file, function_name)

"""
# superpoint_superglue找出的内点更多, 同时目前来看size越大, 找出的内点更多
angle: 0, size: 150, sift, num_matches: 4
angle: 0, size: 150, superpoint_superglue, num_matches: 23
angle: 0, size: 200, sift, num_matches: 6
angle: 0, size: 200, superpoint_superglue, num_matches: 37
angle: 0, size: 250, sift, num_matches: 0
angle: 0, size: 250, superpoint_superglue, num_matches: 56
angle: 0, size: 300, sift, num_matches: 3
angle: 0, size: 300, superpoint_superglue, num_matches: 75
angle: 0, size: 350, sift, num_matches: 2
angle: 0, size: 350, superpoint_superglue, num_matches: 118
angle: 0, size: 400, sift, num_matches: 4
angle: 0, size: 400, superpoint_superglue, num_matches: 139
angle: 0, size: 450, sift, num_matches: 7
angle: 0, size: 450, superpoint_superglue, num_matches: 158
angle: 0, size: 500, sift, num_matches: 6
angle: 0, size: 500, superpoint_superglue, num_matches: 178

angle: 60, size: 150, sift, num_matches: 5
angle: 60, size: 150, superpoint_superglue, num_matches: 21
angle: 60, size: 200, sift, num_matches: 4
angle: 60, size: 200, superpoint_superglue, num_matches: 39
angle: 60, size: 250, sift, num_matches: 2
angle: 60, size: 250, superpoint_superglue, num_matches: 58
angle: 60, size: 300, sift, num_matches: 3
angle: 60, size: 300, superpoint_superglue, num_matches: 64
angle: 60, size: 350, sift, num_matches: 2
angle: 60, size: 350, superpoint_superglue, num_matches: 98
angle: 60, size: 400, sift, num_matches: 3
angle: 60, size: 400, superpoint_superglue, num_matches: 106
angle: 60, size: 450, sift, num_matches: 12
angle: 60, size: 450, superpoint_superglue, num_matches: 126
angle: 60, size: 500, sift, num_matches: 3
angle: 60, size: 500, superpoint_superglue, num_matches: 143
"""