from isaaclab.utils import configclass

from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg as IsaaclabTerrainImporterCfg
from legged_lab.terrain.terrain_importer import TerrainImporter


@configclass
class TerrainImporterCfg(IsaaclabTerrainImporterCfg):
    class_type: type = TerrainImporter
