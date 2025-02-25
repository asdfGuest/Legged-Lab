import isaaclab.terrains as terrain_gen
from legged_lab.terrain import TerrainGeneratorCfg


ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(20.0, 20.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        'pyramid_stairs': terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        'pyramid_stairs_inv': terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        'boxes': terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.44, grid_height_range=(0.025, 0.1), platform_width=1.5
        ),
        'random_rough': terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        'hf_pyramid_slope': terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=1.5, border_width=0.25
        ),
        'hf_pyramid_slope_inv': terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=1.5, border_width=0.25
        ),
    },
)


ROUGH_TERRAIN_SMALL_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(20.0, 20.0),
    border_width=20.0,
    num_rows=5,
    num_cols=6,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        'pyramid_stairs': terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        'pyramid_stairs_inv': terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        'boxes': terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1., grid_width=0.44, grid_height_range=(0.025, 0.1), platform_width=1.5
        ),
        'random_rough': terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1., noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        'hf_pyramid_slope': terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1., slope_range=(0.0, 0.4), platform_width=1.5, border_width=0.25
        ),
        'hf_pyramid_slope_inv': terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1., slope_range=(0.0, 0.4), platform_width=1.5, border_width=0.25
        ),
    },
)
