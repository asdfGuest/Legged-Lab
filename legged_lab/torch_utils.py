
import isaaclab.utils.math as math_utils

import torch as th
from typing import Tuple


def sq_norm(x:th.Tensor, keepdim:bool=False):
    return th.sum(th.square(x), dim=-1, keepdim=keepdim)

def unit(x:th.Tensor):
    return x / (th.norm(x,dim=-1,keepdim=True) + 1e-8)

def vec_ang(x:th.Tensor, y:th.Tensor, keepdim:bool=False):
    cos = (x*y).sum(dim=-1,keepdim=keepdim) / (th.norm(x,dim=-1,keepdim=keepdim) * th.norm(y,dim=-1,keepdim=keepdim) + 1e-8)
    return cos.clip(-1.+1e-8,1.-1e-8).acos()

def dot(x:th.Tensor, y:th.Tensor, keepdim:bool=False):
    return (x*y).sum(dim=-1,keepdim=keepdim)


def get_relative_values(
        pos_a:th.Tensor,
        quat_a:th.Tensor,
        linvel_a:th.Tensor,
        angvel_a:th.Tensor,
        pos_b:th.Tensor,
        quat_b:th.Tensor,
        linvel_b:th.Tensor,
        angvel_b:th.Tensor
    ) ->Tuple[th.Tensor,th.Tensor,th.Tensor,th.Tensor]:
    '''
    compute relative quantities of b respect to a

    Args:
        pos_a:      (n_batch, n_body, 3)
        quat_a:     (n_batch, n_body, 4)
        linvel_a:   (n_batch, n_body, 3)
        angvel_a:   (n_batch, n_body, 3)
        pos_b:      (n_batch, n_body, 3)
        quat_b:     (n_batch, n_body, 4)
        linvel_b:   (n_batch, n_body, 3)
        angvel_b:   (n_batch, n_body, 3)

    Returns:
        - relative position
        - relative quaternion
        - relative linear velocity
        - relative angular velocity
    '''
    pos_a,pos_b = th.broadcast_tensors(pos_a,pos_b)
    quat_a,quat_b = th.broadcast_tensors(quat_a,quat_b)
    linvel_a,linvel_b = th.broadcast_tensors(linvel_a,linvel_b)
    angvel_a,angvel_b = th.broadcast_tensors(angvel_a,angvel_b)

    quat_a_inv = math_utils.quat_inv(quat_a)

    rel_pos = math_utils.quat_rotate(quat_a_inv, pos_b-pos_a)
    rel_quat = math_utils.quat_mul(quat_a_inv, quat_b)

    rel_linvel = math_utils.quat_rotate(quat_a_inv, linvel_b-linvel_a) - th.cross(angvel_a, rel_pos, dim=-1)
    rel_angvel = math_utils.quat_rotate(quat_a_inv, angvel_b-angvel_a)

    return rel_pos, rel_quat, rel_linvel, rel_angvel
