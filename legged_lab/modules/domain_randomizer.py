from isaaclab.managers import SceneEntityCfg, EventTermCfg
from isaaclab.envs.mdp.events import (
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    randomize_actuator_gains,
    push_by_setting_velocity,
)
from legged_lab.utils import get_sampling_table

from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class DomainRandomizerCfg :
    body_names: List[str]
    payload_range: Tuple[float, float]

    static_cof_range: Tuple[float, float]
    dynamic_cof_range: Tuple[float, float]
    restitution_range: Tuple[float, float]
    n_buckets: int

    kp_range: Tuple[float, float]
    kd_range: Tuple[float, float]
    
    push_limit: Tuple[float, float, float, float, float, float]
    '''
    (lin_x, lin_y, lin_z, ang_x, ang_y, ang_z)
    '''
    push_freq: int|None


class DomainRandomizer :
    def __init__(self, env, robot_entity_name:str, cfg:DomainRandomizerCfg) :
        self.env = env
        self.robot_entity_name = robot_entity_name
        self.cfg = cfg

        # randomize rigidbody mass
        mass_entity = SceneEntityCfg(self.robot_entity_name, body_names=self.cfg.body_names)
        mass_entity.resolve(self.env.scene)
        randomize_rigid_body_mass(
            env=env,
            env_ids=None,
            asset_cfg=mass_entity,
            mass_distribution_params=self.cfg.payload_range,
            operation='add',
            distribution='uniform',
            recompute_inertia=True,
        )

        # randomize physics material
        randomize_rigid_body_material(
            cfg=EventTermCfg(
                params={
                    'asset_cfg': SceneEntityCfg(self.robot_entity_name),
                    'static_friction_range': self.cfg.static_cof_range,
                    'dynamic_friction_range': self.cfg.dynamic_cof_range,
                    'restitution_range': self.cfg.restitution_range,
                    'num_buckets': self.cfg.n_buckets,
                    'make_consistent': False,
                },
            ),
            env=self.env,
        ).__call__(
            env=self.env,
            env_ids=None,
            static_friction_range=None,
            dynamic_friction_range=None,
            restitution_range=None,
            num_buckets=self.cfg.n_buckets,
            asset_cfg=None,
        )

        # randomize pd gains
        randomize_actuator_gains(
            env=env,
            env_ids=None,
            asset_cfg=SceneEntityCfg(self.robot_entity_name),
            stiffness_distribution_params=self.cfg.kp_range,
            damping_distribution_params=self.cfg.kd_range,
            operation='scale',
            distribution='uniform',
        )

        # sampling table for random push
        if self.cfg.push_freq is not None :
            self.velocity_range = {
                key:(-val,val) for key, val in zip(['x', 'y', 'z', 'roll', 'pitch', 'yaw'], self.cfg.push_limit)
            }
            self.push_sampler = get_sampling_table(
                elements=self.env.scene.num_envs,
                sample_freq=self.cfg.push_freq,
                device=self.env.device,
            )
            self.cnt = 0
    
    def update(self) :
        if self.cfg.push_freq is not None :
            push_env_ids = self.push_sampler[self.cnt % self.cfg.push_freq]

            if len(push_env_ids) > 0 :
                push_by_setting_velocity(
                    env=self.env,
                    env_ids=push_env_ids,
                    velocity_range=self.velocity_range,
                    asset_cfg=SceneEntityCfg(self.robot_entity_name),
                )
            
            self.cnt += 1
