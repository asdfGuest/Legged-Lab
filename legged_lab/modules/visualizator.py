import torch as th

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VelocityVisualizatorCfg :
    z_pos_offset:float = 0.5
    marker_scale:Tuple[float,float,float] = (0.6, 0.4, 0.4)

    debug_vis:bool = True

    base_velocity_marker_cfg:VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path='/Visuals/VelocityVisualizator/base_velocity_marker',
        markers={
            'arrow': sim_utils.UsdFileCfg(
                usd_path=f'{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd',
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)), # Green
            )
        }
    )
    command_velocity_marker_cfg:VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path='/Visuals/VelocityVisualizator/command_velocity_marker',
        markers={
            'arrow': sim_utils.UsdFileCfg(
                usd_path=f'{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd',
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)), # Blue
            )
        }
    )


class VelocityVisualizator :
    def __init__(self, cfg:VelocityVisualizatorCfg) :
        self.cfg = cfg
        self.base_velocity_marker = VisualizationMarkers(self.cfg.base_velocity_marker_cfg)
        self.command_velocity_marker = VisualizationMarkers(self.cfg.command_velocity_marker_cfg)
        self.visible = self.cfg.debug_vis

    def _compute_marker_transform(self, pos_w:th.Tensor, quat_w:th.Tensor, velocity:th.Tensor) :
        marker_pos = pos_w.clone()
        marker_pos[:, 2] += self.cfg.z_pos_offset

        theta = th.atan2(velocity[:, 1], velocity[:, 0])
        theta_zeros_like = th.zeros_like(theta)
        theta_quat = math_utils.quat_from_euler_xyz(theta_zeros_like, theta_zeros_like, theta)
        marker_quat = math_utils.quat_mul(quat_w, theta_quat)

        marker_scale = th.tensor(self.cfg.marker_scale, dtype=th.float32, device=pos_w.device).repeat(pos_w.shape[0], 1)
        marker_scale[:, 0] *= th.norm(velocity, dim=-1)

        return marker_pos, marker_quat, marker_scale

    def set_set_visibility(self, visible:bool) :
        self.visible = visible
        self.base_velocity_marker.set_visibility(self.visible)
        self.command_velocity_marker.set_visibility(self.visible)
    
    def update(
            self,
            root_pos_w:th.Tensor,
            root_quat_w:th.Tensor,
            root_linvel_b:th.Tensor,
            command:th.Tensor,
        ) :
        '''
        root_pos_w : (n_env, 3)
        root_quat_w : (n_env, 4)
        root_linvel_b : (n_env, 3)
        command : (n_env, 2)
        '''
        if not self.visible :
            return
        
        self.base_velocity_marker.visualize(
            *self._compute_marker_transform(
                pos_w=root_pos_w,
                quat_w=root_quat_w,
                velocity=root_linvel_b
            )
        )
        self.command_velocity_marker.visualize(
            *self._compute_marker_transform(
                pos_w=root_pos_w,
                quat_w=root_quat_w,
                velocity=command
            )
        )
