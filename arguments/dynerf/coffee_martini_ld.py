_base_ = './default.py'
OptimizationParams = dict(
    router_lr_init = 0.00008,
    router_lr_final = 0.000008,
    gaiting_from_iter=2000,
    lamabda_importance = 1
)
ModelHiddenParams = dict(
    sparse=True,
)
