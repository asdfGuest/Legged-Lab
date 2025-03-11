from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor

import isaaclab.utils.math as math_utils
import legged_lab.torch_utils as th_utils

from legged_lab.utils import my_direct_rl_env_step
from legged_lab.modules import (
    GridCommand,
    VelocityVisualizator,
    VelocityVisualizatorCfg,
    DomainRandomizer
)
from legged_lab.base.config import BaseConfig

import torch as th

from _collections_abc import Sequence
from typing import Literal, Tuple, List


class BaseEnv(DirectRLEnv) :
    
    action_tensor:th.Tensor
    '''
    (n_env, 3, n_action)
    '''
    obs_tensor:th.Tensor
    '''
    (n_env, n_history, n_obs)
    '''

    def __init__(self, cfg:BaseConfig) :
        self.n_env = cfg.env.n_env
        self.n_action = cfg.env.n_action
        self.n_obs = cfg.env.n_obs
        self.n_history = cfg.env.n_history
        self.n_privileged_obs = cfg.env.n_privileged_obs

        cfg.scene.scene.num_envs = cfg.env.n_env
        rl_cfg = DirectRLEnvCfg(
            sim = SimulationCfg(
                device=cfg.simulation.device,
                dt=cfg.simulation.physics_dt,
                render_interval=cfg.simulation.decimation,
                physics_material=cfg.simulation.physics_material,
            ),
            decimation = cfg.simulation.decimation,
            episode_length_s = cfg.simulation.physics_dt*cfg.simulation.decimation*cfg.env.episode_length,
            scene = cfg.scene.scene,
            observation_space = (self.n_history,self.n_obs, self.n_privileged_obs),
            action_space = self.n_action,
        )
        super().__init__(rl_cfg)
        self._cfg = cfg

        self.robot:Articulation = self.scene[self._cfg.scene.robot_entity_name]
        self.contact_snsr:ContactSensor = self.scene[self._cfg.scene.contact_sensor_entity_name]
        
        self.init()
    

    def init(self) :
        self.AXIS_X_W = th.tensor([1.0, 0.0, 0.0], dtype=th.float32, device=self.device).repeat(self.n_env, 1)
        self.AXIS_Y_W = th.tensor([0.0, 1.0, 0.0], dtype=th.float32, device=self.device).repeat(self.n_env, 1)
        self.AXIS_Z_W = th.tensor([0.0, 0.0, 1.0], dtype=th.float32, device=self.device).repeat(self.n_env, 1)

        self.extras['Metric'] = {}
        self.extras['Reward'] = {}
        self.extras['Done'] = {}

        class body_id :
            base = self.robot.find_bodies(self._cfg.scene.base_name)[0]
            foot = self.robot.find_bodies(self._cfg.scene.foot_names, preserve_order=True)[0]
        class cont_id :
            base = self.contact_snsr.find_bodies(self._cfg.scene.base_name)[0]
            foot = self.contact_snsr.find_bodies(self._cfg.scene.foot_names, preserve_order=True)[0]
            terminate = self.contact_snsr.find_bodies(self._cfg.scene.contact_terminate_names)[0]
            penalty = self.contact_snsr.find_bodies(self._cfg.scene.contact_penalty_names)[0]
        self.body_id = body_id
        self.cont_id = cont_id
        self.joint_ord = self.robot.find_joints(self._cfg.scene.joint_index_names, preserve_order=True)[0]
        self.joint_ord_inv = th.argsort(th.tensor(self.joint_ord)).tolist()

        self.command_manager = GridCommand(self.n_env, self.device, self._cfg.command)
        self.velocity_visualizator = VelocityVisualizator(
            VelocityVisualizatorCfg(debug_vis=self._cfg.env.debug_vis)
        )
        self.domain_randomizer = DomainRandomizer(self, self._cfg.scene.robot_entity_name, self._cfg.domain_rand)
        
        self.action_tensor = th.zeros(
            size=(self.n_env,3,self.n_action),
            dtype=th.float32,
            device=self.device
        )
        self.obs_tensor_ordered = th.zeros(
            size=(self.n_env,self.n_history,self.n_obs),
            dtype=th.float32,
            device=self.device
        )
        self.filtered_action = th.zeros(
            size=(self.n_env,self.n_action),
            dtype=th.float32,
            device=self.device
        )

        # compute noise vector and scale vector in advance
        def get_noise_and_scale_vector(key_and_dim:List[Tuple[str, int]]) :
            noise_vector = []
            scale_vector = []

            for key, dim in key_and_dim :
                noise = getattr(self._cfg.obs_noise, key)
                scale = getattr(self._cfg.obs_scale, key)
                noise_vector.append(th.ones(size=(1,dim), dtype=th.float32, device=self.device) * noise)
                scale_vector.append(th.ones(size=(1,dim), dtype=th.float32, device=self.device) * scale)
            
            noise_vector = th.cat(noise_vector, dim=-1)
            scale_vector = th.cat(scale_vector, dim=-1)
            return noise_vector, scale_vector
        
        self.obs_noise_vec, self.obs_scale_vec = get_noise_and_scale_vector([
            ('angvel', 3),
            ('gravity', 3),
            ('command', 3),
            ('joint_pos', 12),
            ('joint_vel', 12),
            ('action', 12),
        ])
        self.privileged_obs_noise_vec, self.privileged_obs_scale_vec = get_noise_and_scale_vector([
            ('linvel', 3),
        ])
        if not self._cfg.obs_noise.add_noise :
            self.obs_noise_vec *= 0.0
            self.privileged_obs_noise_vec *= 0.0

        # buffers
        self.follow_sum_buf = th.zeros(size=(self.n_env,), dtype=th.int64, device=self.device)
        self.follow_cnt_buf = th.zeros(size=(self.n_env,), dtype=th.int64, device=self.device)
        self.terrain_level_update_buf = th.zeros(size=(self.n_env,), dtype=th.int64, device=self.device)

        self.root_linvel_b_buff = th.zeros(
            size=(self.n_env,self._cfg.simulation.decimation*self._cfg.terrain_curriculum.linvel_buff_len,3),
            dtype=th.float32,
            device=self.device
        )
    

    @property
    def command(self) :
        return self.command_manager.command[:,:3]
    
    
    def _compute(self, name:Literal['is_contact', 'relative_values'], **kwargs) :
        if name == 'is_contact' :
            return th.any(
                self.contact_snsr.data.net_forces_w_history.norm(dim=-1) > self.contact_snsr.cfg.force_threshold,
                dim=1
            )
        
        elif name == 'relative_values' :
            body_id = kwargs['body_id'] if 'body_id' in kwargs else slice()
            return th_utils.get_relative_values(
                pos_a=self.robot.data.root_pos_w[:,None,:],
                quat_a=self.robot.data.root_quat_w[:,None,:],
                linvel_a=self.robot.data.root_lin_vel_w[:,None,:],
                angvel_a=self.robot.data.root_ang_vel_w[:,None,:],
                pos_b=self.robot.data.body_pos_w[:,body_id],
                quat_b=self.robot.data.body_quat_w[:,body_id],
                linvel_b=self.robot.data.body_lin_vel_w[:,body_id],
                angvel_b=self.robot.data.body_ang_vel_w[:,body_id],
            )
    

    def _pre_physics_step(self, action_ordered: th.Tensor) :
        self.action_tensor = self.action_tensor.roll(shifts=1, dims=1)
        self.action_tensor[:,0,:] = action_ordered[:,self.joint_ord_inv]
        self.action_applied_cnt = 0


    def _apply_action(self) :
        # compute delayed action
        delayed_action = (
            self.action_tensor[:,1,:]
            if self.action_applied_cnt < self._cfg.env.action_delay else
            self.action_tensor[:,0,:]
        )
        self.action_applied_cnt += 1
        # filter action
        self.filtered_action = (
            self.filtered_action * (1.0 - self._cfg.env.lpf_alpha) +
            delayed_action * self._cfg.env.lpf_alpha
        )
        # set pd-controller target
        self.robot.set_joint_position_target(
            self.robot.data.default_joint_pos + self.filtered_action * self._cfg.env.action_scale
        )
        # track reference acc
        if self.action_applied_cnt == self._cfg.env.action_delay + 1 :
            self.ref_acc_sq_norm = th_utils.sq_norm(self.robot.data.joint_acc).mean().item()
    

    def _after_apply_action(self) :
        self.root_linvel_b_buff = self.root_linvel_b_buff.roll(shifts=1, dims=1)
        self.root_linvel_b_buff[:,0,:] = self.robot.data.root_lin_vel_b
    

    def _get_observations(self) :
        root_linvel_b = self.root_linvel_b_buff.mean(dim=1)

        # update command
        self.command_manager.update(self.robot.data.heading_w)
        
        # update velocity visualizator
        self.velocity_visualizator.update(
            root_pos_w=self.robot.data.root_pos_w,
            root_quat_w=self.robot.data.root_quat_w,
            root_linvel_b=root_linvel_b,
            command=self.command[:,:2]
        )

        # update domain randomizer
        self.domain_randomizer.update()
        
        # compute observation
        angvel_term = self.robot.data.root_ang_vel_b
        gravity_term = self.robot.data.projected_gravity_b
        command_term = self.command
        joint_pos_term = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        joint_vel_term = self.robot.data.joint_vel - self.robot.data.default_joint_vel
        action_term = self.action_tensor[:,0,:].clip(
            min=-self._cfg.env.obs_action_clip,
            max=self._cfg.env.obs_action_clip,
        )
        linvel_term = root_linvel_b
        
        obs_t = th.cat([
            angvel_term,
            gravity_term,
            command_term,
            joint_pos_term[:,self.joint_ord],
            joint_vel_term[:,self.joint_ord],
            action_term[:,self.joint_ord],
        ], dim=-1)
        obs_t = (obs_t + (th.rand_like(obs_t) * 2 - 1) * self.obs_noise_vec) * self.obs_scale_vec
        self.obs_tensor_ordered = th.roll(self.obs_tensor_ordered, shifts=1, dims=1)
        self.obs_tensor_ordered[:,0,:] = obs_t

        privileged_obs = linvel_term
        privileged_obs = (privileged_obs + (th.rand_like(privileged_obs) * 2 - 1) * self.privileged_obs_noise_vec) * self.privileged_obs_scale_vec
        
        return {'policy': (self.obs_tensor_ordered.clone(), privileged_obs)}
    
    
    def _get_rewards(self) :
        cfg = self._cfg.reward
        reward_terms = {}
        # velocity tracking
        if cfg.k_track_lin is not None :
            lin_sq_error = th_utils.sq_norm(self.robot.data.root_lin_vel_b[:,:2] - self.command[:,:2])
            reward_terms['track_lin'] = th.exp(-cfg.track_lin_err_scale * lin_sq_error) * cfg.k_track_lin
        if cfg.k_track_ang is not None :
            ang_sq_error = th_utils.sq_norm(self.robot.data.root_ang_vel_b[:,2:] - self.command[:,2:])
            reward_terms['track_ang'] = th.exp(-cfg.track_ang_err_scale * ang_sq_error) * cfg.k_track_ang
        # base motion penalty
        if cfg.k_pnt_lin is not None :
            reward_terms['pnt_lin'] = th_utils.sq_norm(self.robot.data.root_lin_vel_b[:,2:]) * cfg.k_pnt_lin
        if cfg.k_pnt_ang is not None :
            reward_terms['pnt_ang'] = th_utils.sq_norm(self.robot.data.root_ang_vel_b[:,:2]) * cfg.k_pnt_ang
        if cfg.k_orientation is not None :
            reward_terms['orientation'] = th_utils.sq_norm(self.robot.data.projected_gravity_b[:,:2]) * cfg.k_orientation
        if cfg.k_contact is not None :
            is_contact = self._compute('is_contact')
            reward_terms['contact'] = th.sum(is_contact[:,self.cont_id.penalty].to(th.float32), dim=-1) * cfg.k_contact
        # joint motion penalty
        if cfg.k_pos is not None :
            reward_terms['pos'] = th_utils.sq_norm(
                self.robot.data.joint_pos - self.robot.data.default_joint_pos
            ) * cfg.k_pos
        if cfg.k_acc is not None :
            reward_terms['acc'] = th_utils.sq_norm(self.robot.data.joint_acc) * cfg.k_acc
        if cfg.k_trq is not None :
            reward_terms['trq'] = th_utils.sq_norm(self.robot.data.applied_torque) * cfg.k_trq
        if cfg.k_pwr is not None :
            reward_terms['pwr'] = th.sum(self.robot.data.joint_vel.abs() * self.robot.data.applied_torque.abs(), dim=-1) * cfg.k_pwr
        if cfg.k_action_rate is not None :
            diff = self.action_tensor[:,0,:] - self.action_tensor[:,1,:]
            reward_terms['action_rate'] = th_utils.sq_norm(diff) * cfg.k_action_rate
        if cfg.k_action_smooth is not None :
            diff = 2 * self.action_tensor[:,1,:] - self.action_tensor[:,0,:] - self.action_tensor[:,2,:]
            reward_terms['action_smooth'] = th_utils.sq_norm(diff) * cfg.k_action_smooth
        # foot motion penalty
        def oblique_projection(q:th.Tensor, n:th.Tensor, d:th.Tensor, x:th.Tensor) :
            '''
            Args:
                q:  (n_batch, 3)
                n:  (n_batch, 3)
                d:  (n_batch, 3)
                x:  (n_batch, n_point, 3)
            '''
            k = th_utils.dot(q[:,None,:] - x, n[:,None,:]) / th_utils.dot(d, n, keepdim=True) #(n_batch, n_point)
            x_proj = x + k[:,:,None] * d[:,None,:] #(n_batch, n_point, 3)
            return x_proj
        
        foot_pos_w = self.robot.data.body_pos_w[:,self.body_id.foot,:]
        foot_pos_proj = oblique_projection(
            q=self.robot.data.root_pos_w,
            n=math_utils.quat_rotate(self.robot.data.root_link_quat_w, self.AXIS_Z_W),
            d=self.AXIS_Z_W,
            x=foot_pos_w
        )
        foot_err_clip = (2.0 * cfg.target_mean_foot_height)**2
        
        if cfg.k_foot_height is not None :
            mean_foot_height = th.mean(foot_pos_w[:,:,2] - foot_pos_proj[:,:,2], dim=-1).unsqueeze(-1)
            reward_terms['foot_height'] = th_utils.sq_norm(
                mean_foot_height - cfg.target_mean_foot_height
            ).clip(max=foot_err_clip) * cfg.k_foot_height
        if cfg.k_foot_clear is not None :
            foot_pos_z_err = th_utils.sq_norm(
                foot_pos_proj[:,:,2:] + cfg.target_lift_foot_height - foot_pos_w[:,:,2:]
            ).clip(max=foot_err_clip)
            foot_vel_xy = th.norm(self.robot.data.body_lin_vel_w[:,self.body_id.foot,:2], dim=-1)
            reward_terms['foot_clear'] = th.sum(foot_pos_z_err * foot_vel_xy, dim=-1) * cfg.k_foot_clear
        # simulation stability
        if cfg.k_torque_limit is not None :
            reward_terms['torque_limit'] = th.sum(
                (self.robot.data.applied_torque.abs() - cfg.torque_threshold).clip(min=0.0)
            ,dim=-1) * cfg.k_torque_limit
        
        # compute total reward
        total_reward:th.Tensor = sum(reward_terms.values())
        total_reward.clip_(min=cfg.minimum_reward)
        if cfg.non_negative_mean :
            add = -total_reward.mean().clip(max=0.0)
            total_reward += add
            reward_terms['add'] = add
        
        self.extras['Reward'] = {k:th.mean(v).item() for k, v in reward_terms.items()}

        # compute following rate
        root_linvel_b = self.root_linvel_b_buff.mean(dim=1)
        
        xy_diff_norm = th.norm(root_linvel_b[:,:2] - self.command[:,:2], dim=-1)
        xy_cmd_norm = th.norm(self.command[:,:2], dim=-1)
        is_following = xy_diff_norm < xy_cmd_norm * self._cfg.terrain_curriculum.following_boundary

        self.follow_sum_buf += is_following * self.command_manager.grid_on
        self.follow_cnt_buf += self.command_manager.grid_on

        follow_rate = self.follow_sum_buf / (self.follow_cnt_buf + 1e-6)
        # update terrain curriculum based on following rate
        on_update = self.follow_cnt_buf >= self._cfg.terrain_curriculum.buffer_length

        self.terrain_level_update_buf[on_update*(follow_rate<self._cfg.terrain_curriculum.following_rate_threshold[0])] -= 1
        self.terrain_level_update_buf[on_update*(follow_rate>self._cfg.terrain_curriculum.following_rate_threshold[1])] += 1
        
        self.follow_sum_buf[on_update] = 0
        self.follow_cnt_buf[on_update] = 0
        # logging
        self.extras['Metric']['following_rate'] = is_following.to(th.float32).mean().item()
        self.extras['Metric']['terrain_level'] = self.scene.terrain.terrain_levels.to(th.float32).mean().item()
        
        cur_acc_sq_norm = th_utils.sq_norm(self.robot.data.joint_acc).mean().item()
        self.extras['Metric']['acc_ratio'] = min(cur_acc_sq_norm / (self.ref_acc_sq_norm + 1e-8), 4.0)
        
        return total_reward
    

    def _get_dones(self) :
        is_contact = self._compute('is_contact')
        axis_z_b = math_utils.quat_rotate(self.robot.data.root_link_quat_w, self.AXIS_Z_W)

        terminated = (
            th.any(is_contact[:,self.cont_id.terminate], dim=-1) |
            (th_utils.vec_ang(self.AXIS_Z_W, axis_z_b) > self._cfg.env.terminate_ang)
        )
        truncated = self.episode_length_buf >= self.max_episode_length

        self.extras['Done']['termination'] = th.sum(terminated).item()
        self.extras['Done']['truncation'] = th.sum(truncated).item()

        return terminated, truncated
    
    
    def _update_terrain(self, env_ids:Sequence[int]) :
        terrain_level_update_buf = self.terrain_level_update_buf[env_ids]

        self.scene.terrain.update_env_origins(
            env_ids,
            move_up=terrain_level_update_buf > 0,
            move_down=terrain_level_update_buf < 0,
        )
        self.terrain_level_update_buf[env_ids] = 0
    
    
    def _reset_idx(self, env_ids:Sequence[int]) :
        super()._reset_idx(env_ids)
        
        self.action_tensor[env_ids] = 0.0
        self.obs_tensor_ordered[env_ids] = 0.0
        self.filtered_action[env_ids] = 0.0
        self.root_linvel_b_buff[env_ids] = 0.0

        if self.n_env > 1 and len(env_ids) == self.n_env :
            self.episode_length_buf[:] = th.randint(low=0, high=self.max_episode_length, size=(self.n_env,), device=self.device)
        
        self._update_terrain(env_ids)

        # reset root state
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:,:3] += self.scene.terrain.env_origins[env_ids]
        self.robot.write_root_state_to_sim(root_state, env_ids)
        # reset joint state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    

    def step(self, action) :
        return my_direct_rl_env_step(self, action)
