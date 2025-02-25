import isaaclab.assets as assets
import isaaclab.sensors as sensors
import isaaclab.sim as sim_utils
import legged_lab.terrain as terrain

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_A1_CFG


@configclass
class RoughSceneCfg(InteractiveSceneCfg) :
    env_spacing = 2.5

    # light
    sky_light = assets.AssetBaseCfg(
        prim_path='/World/SkyLight',
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f'{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr',
        ),
    )
    
    # terrain
    terrain = terrain.TerrainImporterCfg(
        prim_path='/World/ground',
        terrain_type='generator',
        terrain_generator=terrain.ROUGH_TERRAIN_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode='multiply',
            restitution_combine_mode='multiply',
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f'{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl',
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
    )
    
    # sensors
    contact_sensor = sensors.ContactSensorCfg(
        prim_path='{ENV_REGEX_NS}/Robot/.*',
        update_period=1/200,
        history_length=3,
        track_air_time=True
    )
    
    # robot
    robot = UNITREE_A1_CFG
    robot.prim_path = '{ENV_REGEX_NS}/Robot'


@configclass
class FlatSceneCfg(RoughSceneCfg) :
    terrain = terrain.TerrainImporterCfg(
        prim_path='/World/ground',
        terrain_type='plane',
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode='multiply',
            restitution_combine_mode='multiply',
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
    )
