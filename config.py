class Args:
    data_root = '/workspace/personal/classification/dataset'
    train_list = '/workspace/personal/classification/dataset/train.txt'
    val_list = '/workspace/personal/classification/dataset/test.txt'
    arch = 'resnet50' # 网络架构, resnet50或se_resnet50
    num_classes = 9 # 类别数
    batch_size = 64
    lr = 0.001
    momentum = 0.9
    weight_decay = 1e-5
    warm_up = 100 # lr warm_up step
    epoch = 50
    start_epoch = 0
    num_workers = 4
    print_freq = 5
    gpus = '0,1' # 使用的GPU, 例如0,1,2,3
    checkpoint = None
    checkpoint_dir = './checkpoint_mdf_rgb'
