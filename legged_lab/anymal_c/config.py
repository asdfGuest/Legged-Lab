from legged_lab.base.config import BaseConfig
from legged_lab.anymal_c.scene import RoughSceneCfg
from isaaclab.utils import configclass

from legged_lab.terrain import ROUGH_TERRAIN_SMALL_CFG

_BASE = 'base'
_THIGH = ['LF_THIGH', 'LH_THIGH', 'RF_THIGH', 'RH_THIGH']
_FOOT = ['LF_FOOT', 'LH_FOOT', 'RF_FOOT', 'RH_FOOT']


@configclass
class AnymalCCfg(BaseConfig) :
    class scene(BaseConfig.scene) :
        scene = RoughSceneCfg()
        base_name = _BASE
        foot_names = _FOOT
        contact_terminate_names = [_BASE]
        contact_penalty_names = _THIGH
        robot_entity_name = 'robot'
        contact_sensor_entity_name = 'contact_sensor'
    class env(BaseConfig.env) :
        action_scale = 0.5
    class reward(BaseConfig.reward) :
        target_mean_foot_height = -0.5
        target_lift_foot_height = -0.34
        k_torque_limit = None
    class domain_rand(BaseConfig.domain_rand) :
        body_names = [_BASE]


@configclass
class AnymalCPlayCfg(AnymalCCfg) :
    class simulation(AnymalCCfg.simulation) :
        device = 'cuda:0'
    class env(AnymalCCfg.env) :
        n_env = 20
        debug_vis = True
    class obs_noise(AnymalCCfg.obs_noise) :
        add_noise = False
    
    def __post_init__(self) :
        super().__post_init__()
        self.scene.scene.terrain.terrain_generator = ROUGH_TERRAIN_SMALL_CFG
