class FLAGS(object):
    # 路径文件夹
    evalDir = 'evalResult'
    cacheDir = 'cache'
    weightDir = 'model_weights'
    logDir = 'model_logs'

    # 模型属性
    network_def = 'cpm_hand_model'
    model_name = 'cpm_hand'
    # network_def = 'res_encoder_model'
    # model_name = 'res_encoder_hand'
    data_set = 'msra15'
    INPUT_TYPE = 'DEPTH'
    input_size = 128
    joints = 21

    # CPM 模型
    stages = 5

    # res_encoder 模型
    embed = 30
    stacks = 4

    # 训练参数
    batch_size = 64
    init_lr = 1e-3
    weight_decay = 1e-3
    # weight_decay = 0
    lr_decay_rate = 0.92
    lr_decay_step = 1000
    training_iters = 30000
    verbose_iters = 100
    model_save_iters = 5000
    validation_iters = 2000
    pretrained_model = ''

    # 测试参数
    test_iters = 30000

