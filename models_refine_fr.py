
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
#        assert(opt.dataset_mode == 'aligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'cycle_gan_sr':
        from .region2_wu_reslearn_dual_att_stage_1006_bn_revised_12blocks_twotogether import CycleGANModel
        model = CycleGANModel()
        else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
