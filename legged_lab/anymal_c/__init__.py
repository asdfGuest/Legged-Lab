import legged_lab.task as task

from legged_lab.base.env import BaseEnv
from legged_lab.anymal_c.config import (
    AnymalCCfg,
    AnymalCPlayCfg,
)

task.register(
    'AnymalC',
    BaseEnv,
    AnymalCCfg,
)
task.register(
    'AnymalC-Play',
    BaseEnv,
    AnymalCPlayCfg,
)
