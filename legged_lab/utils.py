import torch as th
from typing import List, Tuple


def get_linear_fn(points:List[Tuple[float,float]]) :
    import bisect
    
    points = sorted(points, key=lambda point: point[0])
    x_list = [x for x,_ in points]
    y_list = [y for _,y in points]

    if len(x_list) < 2 :
        raise Exception('At least two points must be provided.')
    if len(set(x_list)) < len(x_list) :
        raise Exception('Multiple points have the same x-coordinate.')
    
    def f(x:float) :
        if x <= x_list[0] :
            return y_list[0]
        elif x >= x_list[-1] :
            return y_list[-1]
        else :
            idx = bisect.bisect_left(x_list, x) - 1
            ratio = (x - x_list[idx]) / (x_list[idx + 1] - x_list[idx])
            return y_list[idx] * (1 - ratio) + y_list[idx + 1] * ratio
    
    return f


def get_sampling_table(elements:List[int]|int, sample_freq:int, device:th.device, decimation:int=1, shuffle:bool=True) -> List[th.Tensor]:
    if sample_freq % decimation != 0 :
        raise Exception('sample_freq must divided by decimation')
    
    if isinstance(elements, int) :
        elements = list(range(elements))
    
    if shuffle :
        mixed_idx = th.randperm(len(elements))
        elements = th.tensor(elements)[mixed_idx].tolist()
    
    indices = [[] for _ in range(sample_freq)]
    for idx, val in enumerate(elements) :
        indices[(idx%(sample_freq//decimation))*decimation].append(val)
    
    for idx in range(sample_freq) :
        indices[idx].sort()
        indices[idx] = th.tensor(indices[idx], dtype=th.int64, device=device)
    return indices


from isaaclab.envs import DirectRLEnv
from isaaclab.envs.common import VecEnvStepReturn

def my_direct_rl_env_step(self:DirectRLEnv, action: th.Tensor) -> VecEnvStepReturn:
    """Execute one time-step of the environment's dynamics.

    The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
    lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
    independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
    and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
    time-step is computed as the product of the two.

    This function performs the following steps:

    1. Pre-process the actions before stepping through the physics.
    2. Apply the actions to the simulator and step through the physics in a decimated manner.
    3. Compute the reward and done signals.
    4. Reset environments that have terminated or reached the maximum episode length.
    5. Apply interval events if they are enabled.
    6. Compute observations.

    Args:
        action: The actions to apply on the environment. Shape is (num_envs, action_dim).

    Returns:
        A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
    """
    action = action.to(self.device)
    # add action noise
    if self.cfg.action_noise_model:
        action = self._action_noise_model.apply(action)

    # process actions
    self._pre_physics_step(action)

    # check if we need to do rendering within the physics loop
    # note: checked here once to avoid multiple checks within the loop
    is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

    # perform physics stepping
    for _ in range(self.cfg.decimation):
        self._sim_step_counter += 1
        # set actions into buffers
        self._apply_action()
        # set actions into simulator
        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
            self.sim.render()
        # update buffers at sim dt
        self.scene.update(dt=self.physics_dt)
        
        self._after_apply_action()

    # post-step:
    # -- update env counters (used for curriculum generation)
    self.episode_length_buf += 1  # step in current episode (per env)
    self.common_step_counter += 1  # total step (common for all envs)

    self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
    self.reset_buf = self.reset_terminated | self.reset_time_outs
    self.reward_buf = self._get_rewards()

    # -- reset envs that terminated/timed-out and log the episode information
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
        self._reset_idx(reset_env_ids)
        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()
        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

    # post-step: step interval event
    if self.cfg.events:
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

    # update observations
    self.obs_buf = self._get_observations()

    # add observation noise
    # note: we apply no noise to the state space (since it is used for critic networks)
    if self.cfg.observation_noise_model:
        self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

    # return observations, rewards, resets and extras
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
