import legged_lab.task as task

from legged_lab.base.env import BaseEnv
from legged_lab.go2.config import (
    Go2Cfg,
    Go2PlayCfg,
)

task.register(
    'Go2',
    BaseEnv,
    Go2Cfg,
)
task.register(
    'Go2-Play',
    BaseEnv,
    Go2PlayCfg,
)
