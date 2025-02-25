from legged_lab.base.config import BaseConfig
from legged_lab.a1.scene import RoughSceneCfg
from isaaclab.utils import configclass

from legged_lab.terrain import ROUGH_TERRAIN_SMALL_CFG

_BASE = 'trunk'
_FOOT = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']


@configclass
class A1Cfg(BaseConfig) :
    class scene(BaseConfig.scene) :
        scene = RoughSceneCfg()
        base_name = _BASE
        foot_names = _FOOT
        contact_terminate_names = [_BASE]
        contact_penalty_names = []
        robot_entity_name = 'robot'
        contact_sensor_entity_name = 'contact_sensor'
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
