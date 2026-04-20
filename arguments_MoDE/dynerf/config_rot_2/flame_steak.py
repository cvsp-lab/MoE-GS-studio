_base_ = './default.py'
OptimizationParams = dict(
    batch_size=2,
    gaiting_from_iter=3_000,
)
ModelHiddenParams = dict(
    MoE_mean = True,
    MoE_rotation = True,
    MoE_opacity = False,
    control_num = 5
)