from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import RigidBodyMaterialCfg

from legged_lab.modules import GridCommandCfg, DomainRandomizerCfg

from isaaclab.utils import configclass
from dataclasses import MISSING


@configclass
class BaseConfig :
    class simulation :
        device = 'cuda:0'
        physics_dt = 1 / 200
        decimation = 4

        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode='multiply',
            restitution_combine_mode='multiply',
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    
    class scene :
        scene:InteractiveSceneCfg = MISSING
        base_name:str = MISSING
        foot_names:list[str] = MISSING
        contact_terminate_names:list[str] = MISSING
        contact_penalty_names:list[str] = MISSING
        
        robot_entity_name:str = MISSING
        contact_sensor_entity_name:str = MISSING
    
    class env :
        episode_length = 1000 # 20s
        
        n_env = 4096
        n_action = 12
        n_obs = 45
        n_history = 5
        n_privileged_obs = 3
        
        action_scale = 0.25
        obs_action_clip = 10.0
        action_delay = 2
        '''
        Modifying this value may require modifying reward configuration. \\
        Reward of joint torque and acceleration are affected by this setting.
        '''
        lpf_alpha = 0.5

        terminate_ang = 1.4 #80 degree
        
        debug_vis = False
    
    class command(GridCommandCfg) :
        x_range = (-1.0, 1.0)
        y_range = (-1.0, 1.0)
        z_range = (-1.0, 1.0)
        x_grid_num = 8
        y_grid_num = 8
        idle_xy_norm = 0.2
        idle_z_norm = 0.2
        heading = True
        heading_stiffness = 0.5
        sample_freq = 500
    
    class terrain_curriculum :
        following_boundary = 0.35
        following_rate_threshold = (0.5, 0.7)
        buffer_length = 500
    
    class reward :
        # velocity tracking
        k_track_lin = 1.0
        k_track_ang = 0.5
        track_lin_err_scale = 4.0
        track_ang_err_scale = 4.0
        # base motion penalty
        k_pnt_lin = -2.0
        k_pnt_ang = -0.05
        k_orientation = -0.2
        k_contact = -0.5
        # joint motion penalty
        k_pos = None
        k_acc = -1.0e-7
        k_trq = None
        k_pwr = None
        k_action_rate = -0.005
        k_action_smooth = -0.005
        # foot motion penalty
        k_foot_height = -2.5
        '''
        This term is alternative of base height reward.
        '''
        target_mean_foot_height = MISSING
        k_foot_clear = -0.5
        target_lift_foot_height = MISSING
        # simulation stability
        k_torque_limit = -0.5
        torque_threshold = 20.0
        
        minimum_reward = -10.0
        non_negative_mean = True
    
    class domain_rand(DomainRandomizerCfg) :
        body_names = MISSING
        payload_range = (-1.0, 3.0)
        
        static_cof_range = (0.4, 1.25)
        dynamic_cof_range = (0.2, 1.0)
        restitution_range = (0.0, 0.2)
        n_buckets = 4096

        kp_range = (0.85, 1.15)
        kd_range = (0.85, 1.15)
        
        push_limit = (1.0, 1.0, 0.5, 0.5, 0.5, 0.5)
        push_freq = 500
    
    class obs_scale :
        # obs
        angvel = 0.25
        gravity = 1.0
        command = 1.0
        joint_pos = 1.0
        joint_vel = 0.05
        action = 1.0
        # privileged obs
        linvel = 2.0
    
    class obs_noise :
        add_noise = True
        # obs
        angvel = 0.2
        gravity = 0.05
        command = 0.0
        joint_pos = 0.01
        joint_vel = 1.5
        action = 0.0
        # privileged obs
        linvel = 0.1
