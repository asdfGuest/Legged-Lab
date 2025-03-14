from legged_lab.base.config import BaseConfig
from legged_lab.a1.scene import RoughSceneCfg
from isaaclab.utils import configclass

from legged_lab.terrain import ROUGH_TERRAIN_SMALL_CFG

_BASE = 'trunk'
_THIGH = ['FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh']
_CALF = ['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
_FOOT = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']


@configclass
class A1Cfg(BaseConfig) :
    class scene(BaseConfig.scene) :
        scene = RoughSceneCfg()
        base_name = _BASE
        foot_names = _FOOT
        contact_terminate_names = []
        contact_penalty_names = [_BASE]
        robot_entity_name = 'robot'
        contact_sensor_entity_name = 'contact_sensor'
        joint_index_names = [
            'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
            'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
            'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint',
        ]
    class reward(BaseConfig.reward) :
        target_mean_foot_height = -0.27
        target_lift_foot_height = -0.18
    class domain_rand(BaseConfig.domain_rand) :
        body_names = [_BASE]


@configclass
class A1PlayCfg(A1Cfg) :
    class simulation(A1Cfg.simulation) :
        device = 'cuda:0'
    class env(A1Cfg.env) :
        n_env = 20
        debug_vis = True
    class obs_noise(A1Cfg.obs_noise) :
        add_noise = False
    
    def __post_init__(self) :
        super().__post_init__()
        self.scene.scene.terrain.terrain_generator = ROUGH_TERRAIN_SMALL_CFG
