import legged_lab.task as task

from legged_lab.base.env import BaseEnv
from legged_lab.a1.config import (
    A1Cfg,
    A1PlayCfg,
)

task.register(
    'A1',
    BaseEnv,
    A1Cfg,
)
task.register(
    'A1-Play',
    BaseEnv,
    A1PlayCfg,
)
