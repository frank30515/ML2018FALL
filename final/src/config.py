class DefaultConfigs(object):
    train_data = "./train/"
    test_data = "./test/"   
    weights = "./checkpoints/"
    best_models = "./checkpoints/best_models/"
    model_name = "inception"
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4
    lr = 0.03
    batch_size = 31
    epochs = 50

config = DefaultConfigs()
