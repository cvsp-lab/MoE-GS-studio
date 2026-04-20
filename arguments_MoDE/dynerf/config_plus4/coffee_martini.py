_base_ = './default.py'
OptimizationParams = dict(
    gaiting_from_iter=12_000,
)
ModelHiddenParams = dict(
    MoE_mean = True,
    MoE_rotation = True,
    MoE_opacity = False,
    control_num = 3
)