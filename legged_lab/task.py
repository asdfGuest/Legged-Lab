from legged_lab.base.env import BaseEnv
from legged_lab.base.config import BaseConfig
from typing import Type, Tuple, Dict


_task_table:Dict[str,Tuple[Type[BaseEnv],Type[BaseConfig]]] = {}


def register(env_name:str, env_cls:Type[BaseEnv], cfg_cls:Type[BaseConfig]) :
    _task_table[env_name] = (env_cls, cfg_cls)


def get(env_name:str) :
    return _task_table[env_name]


def make(env_name:str) :
    env_cls, cfg_cls = _task_table[env_name]
    return env_cls(cfg_cls())
