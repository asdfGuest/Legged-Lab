import torch as th
from legged_lab.utils import get_sampling_table
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GridCommandCfg :
    x_range:Tuple[float,float]
    y_range:Tuple[float,float]
    z_range:Tuple[float,float]
    x_grid_num:int
    y_grid_num:int
    idle_xy_norm:float
    idle_z_norm:float
    heading:bool
    heading_stiffness:float
    sample_freq:int


class GridCommand :

    grid_low:th.Tensor
    '''
    float32(n_grid, 2)
    '''
    grid_high:th.Tensor
    '''
    float32(n_grid, 2)
    '''
    _command:th.Tensor
    '''
    float32(n_env, 4)
    '''
    command:th.Tensor
    '''
    float32(n_env, 4)
    '''
    _grid_ids:th.Tensor
    '''
    int64(n_env,)
    '''
    grid_ids:th.Tensor
    '''
    int64(n_env,)
    '''
    grid_on:th.Tensor
    '''
    bool(n_env,)
    - True of grid_on means that corresponding grid_ids is using original _grid_ids.
    - False means that grid_ids is using new sampled value.
    '''
    grid_eps:th.Tensor
    '''
    float32(n_env, 2)
    '''
    is_xy_idle:th.Tensor
    '''
    bool(n_env,)
    '''
    is_z_idle:th.Tensor
    '''
    bool(n_env,)
    '''

    def __init__(self, n_env:int, device:th.device, cfg:GridCommandCfg) :
        self.n_env = n_env
        self.device = device
        self.cfg = cfg

        self.ALL_ENV_IDS = th.arange(self.n_env, dtype=th.int64, device=self.device)
        self.sampler = get_sampling_table(self.n_env, self.cfg.sample_freq, self.device)
        self.n_grid = self.cfg.x_grid_num * self.cfg.y_grid_num
        self.cnt = 0
        
        # compute grid_low/grid_high
        x_range, y_range = self.cfg.x_range, self.cfg.y_range
        x_grid_size = (x_range[1] - x_range[0]) / self.cfg.x_grid_num
        y_grid_size = (y_range[1] - y_range[0]) / self.cfg.y_grid_num
        
        self.grid_low, self.grid_high = [], []
        for i in range(self.cfg.x_grid_num) :
            for j in range(self.cfg.y_grid_num) :
                x_low = x_range[0] + x_grid_size * i
                x_high = x_range[0] + x_grid_size * (i + 1)
                y_low = y_range[0] + y_grid_size * j
                y_high = y_range[0] + y_grid_size * (j + 1)
                self.grid_low.append([x_low, y_low])
                self.grid_high.append([x_high, y_high])
        
        self.grid_low = th.tensor(self.grid_low, dtype=th.float32, device=self.device)
        self.grid_high = th.tensor(self.grid_high, dtype=th.float32, device=self.device)

        # sample initial command
        self._grid_ids = (th.arange(self.n_env) % self.n_grid)[th.randperm(self.n_env)].to(self.device)
        self.grid_on = (th.arange(self.n_env) % 2).to(th.bool)[th.randperm(self.n_env)].to(self.device)
        self.grid_ids = self._grid_ids.clone()
        self.grid_eps = th.empty((self.n_env,2), dtype=th.float32, device=self.device)
        self._command = th.empty((self.n_env,4), dtype=th.float32, device=self.device)
        self.command = th.empty((self.n_env,4), dtype=th.float32, device=self.device)
        self.is_xy_idle = th.empty((self.n_env,), dtype=th.bool, device=self.device)
        self.is_z_idle = th.empty((self.n_env,), dtype=th.bool, device=self.device)

        self._sample(self.ALL_ENV_IDS)
    
    
    def _sample(self, env_ids:th.Tensor) :
        n_env = len(env_ids)
        grid_on = self.grid_on[env_ids]
        grid_ids = self._grid_ids[env_ids]
        grid_ids[grid_on] = th.randint(self.n_grid, size=(grid_on.sum().item(),), device=self.device)
        grid_eps = th.rand(size=(n_env,2), dtype=th.float32, device=self.device)

        _xy_cmd = self.grid_low[grid_ids] * (1-grid_eps) + self.grid_high[grid_ids] * grid_eps
        _z_eps = th.rand(size=(n_env,1), dtype=th.float32, device=self.device)
        _z_cmd = (self.cfg.z_range[1] - self.cfg.z_range[0]) * _z_eps + self.cfg.z_range[0]
        _w_eps = th.rand(size=(n_env,1), dtype=th.float32, device=self.device)
        _w_cmd = (2 * th.pi) * _w_eps - th.pi
        _command = th.cat([
            _xy_cmd,
            _z_cmd,
            _w_cmd,
        ], dim=-1)
        
        is_xy_idle = _command[:,0:2].norm(dim=-1) < self.cfg.idle_xy_norm
        is_z_idle = _command[:,2].abs() < self.cfg.idle_z_norm
        command = _command.clone()
        command[is_xy_idle,0:2] = 0.0
        command[is_z_idle,2] = 0.0

        self.grid_on[env_ids] = ~grid_on
        self.grid_ids[env_ids] = grid_ids
        self.grid_eps[env_ids] = grid_eps
        self._command[env_ids] = _command
        self.is_xy_idle[env_ids] = is_xy_idle
        self.is_z_idle[env_ids] = is_z_idle
        self.command[env_ids] = command
    
    
    def update(self, heading:th.Tensor=None) :
        '''
        heading : (n_env,)
        '''

        # sample new commands
        self._sample(self.sampler[self.cnt % self.cfg.sample_freq])

        # compute z command based on heading
        if self.cfg.heading :
            diff = (self._command[:,3] - heading) % (th.pi * 2)
            diff -= (th.pi * 2) * (diff > th.pi)
            self._command[:,2] = th.clip(diff * self.cfg.heading_stiffness, *self.cfg.z_range)

            self.command[:,2] = self._command[:,2]
            self.is_z_idle = self._command[:,2].abs() < self.cfg.idle_z_norm
            self.command[self.is_z_idle,2] = 0.0
        
        self.cnt += 1
