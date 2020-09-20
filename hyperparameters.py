class hyperparameters:
    '''
    하이퍼파라미터를 쉽게 관리하기 위한 .pt
    '''
    def __init__(self, preset):
        self.SEED = 1602

        self.INPUT_SIZE = 28 * 28
        self.OUTPUT_SIZE = 10
        self.PRESET = preset

        if preset == 0:
            self.BATCH_SIZE = 100
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 3
            self.HIDDEN_L_SIZE = (300, 200)
            self.TRAINING_EPOCH = 30
            self.WEIGHT_INIT = None
            self.OPTIMIZER = None
            self.WEIGHT_DECAY = 0
            self.DROPOUT = 0
        elif preset == 1:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 3
            self.HIDDEN_L_SIZE = (300, 200)
            self.TRAINING_EPOCH = 30
            self.WEIGHT_INIT = None
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0
            self.DROPOUT = 0
        elif preset == 2:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 4
            self.HIDDEN_L_SIZE = (200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = None
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0
            self.DROPOUT = 0
        elif preset == 3:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 4
            self.HIDDEN_L_SIZE = (600, 600, 800)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = None
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0
            self.DROPOUT = 0
        elif preset == 4:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 4
            self.HIDDEN_L_SIZE = (200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = None
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0
            self.DROPOUT = 0
        elif preset == 5:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 4
            self.HIDDEN_L_SIZE = (200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = "he"
            self.OPTIMIZER = "adadelta"
            self.WEIGHT_DECAY = 0
            self.DROPOUT = 0
        elif preset == 6:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 4
            self.HIDDEN_L_SIZE = (200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = "he"
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.01
            self.DROPOUT = 0
        elif preset == 7:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.01
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 4
            self.HIDDEN_L_SIZE = (200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = "he"
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.01
            self.DROPOUT = 0
        elif preset == 8:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.01
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 4
            self.HIDDEN_L_SIZE = (200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = "he"
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.01
            self.DROPOUT = 0.2
        elif preset == 9:
            self.BATCH_SIZE = 100
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 3
            self.HIDDEN_L_SIZE = (300, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = None
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0
            self.DROPOUT = 0
        elif preset == 10:
            self.BATCH_SIZE = 100
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 4
            self.HIDDEN_L_SIZE = (300, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = None
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0
            self.DROPOUT = 0
        elif preset == 11:
            self.BATCH_SIZE = 100
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = None
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0
            self.DROPOUT = 0
        elif preset == 12:
            self.BATCH_SIZE = 100
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 6
            self.HIDDEN_L_SIZE = (300, 200, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = None
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0
            self.DROPOUT = 0
        elif preset == 13:
            self.BATCH_SIZE = 100
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 7
            self.HIDDEN_L_SIZE = (300, 200, 200, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = None
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0
            self.DROPOUT = 0
        elif preset == 14:
            self.BATCH_SIZE = 100
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0
            self.DROPOUT = 0
        elif preset == 15:
            self.BATCH_SIZE = 100
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'xe'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0
            self.DROPOUT = 0
        elif preset == 16:
            self.BATCH_SIZE = 100
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.1
            self.DROPOUT = 0
        elif preset == 17:
            self.BATCH_SIZE = 100
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.01
            self.DROPOUT = 0
        elif preset == 18:
            self.BATCH_SIZE = 100
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.001
            self.DROPOUT = 0
        elif preset == 19:
            self.BATCH_SIZE = 100
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.001
            self.DROPOUT = 0.2
        elif preset == 20:
            self.BATCH_SIZE = 100
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.001
            self.DROPOUT = 0.5
        elif preset == 21:
            self.BATCH_SIZE = 100
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.001
            self.DROPOUT = 0.8
        elif preset == 22:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.001
            self.DROPOUT = 0.2
        elif preset == 23:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.001
            self.DROPOUT = 0.15
        elif preset == 24:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.001
            self.DROPOUT = 0.25
        elif preset == 25:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.001
            self.DROPOUT = 0.3
        elif preset == 26:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adadelta"
            self.WEIGHT_DECAY = 0.001
            self.DROPOUT = 0.2
        elif preset == 27:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.01
            self.DROPOUT = 0.2
        elif preset == 28:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.1
            self.DROPOUT = 0.2
        elif preset == 29:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 1
            self.DROPOUT = 0.2
        elif preset == 30:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 10
            self.DROPOUT = 0.2
        elif preset == 31:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.0001
            self.DROPOUT = 0.2
        elif preset == 32:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.01
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.0001
            self.DROPOUT = 0.2
        elif preset == 33:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.1
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.0001
            self.DROPOUT = 0.2
        elif preset == 34:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 1
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.0001
            self.DROPOUT = 0.2
        elif preset == 35:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.0001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 5
            self.HIDDEN_L_SIZE = (300, 200, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.0001
            self.DROPOUT = 0.2
        elif preset == 36:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 4
            self.HIDDEN_L_SIZE = (300, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.0001
            self.DROPOUT = 0.2
        elif preset == 37:
            self.BATCH_SIZE = 200
            self.LEARNING_RATE = 0.001
            self.ACT_FUNC = "relu"
            self.N_OF_HIDDEN_L = 4
            self.HIDDEN_L_SIZE = (300, 200, 200)
            self.TRAINING_EPOCH = 100
            self.WEIGHT_INIT = 'he'
            self.OPTIMIZER = "adam"
            self.WEIGHT_DECAY = 0.01
            self.DROPOUT = 0.2

    def pprint(self):
        print(">> Hyperparameters")
        print(f"= Preset                 : {self.PRESET}")
        print(f"= Batch_size             : {self.BATCH_SIZE}")
        print(f"= Learning_rate          : {self.LEARNING_RATE}")
        print(f"= Act_func               : {self.ACT_FUNC}")
        print(f"= #_of_hidden_l          : {self.N_OF_HIDDEN_L}")
        print(f"= Hidden_l_size          : {self.HIDDEN_L_SIZE}")
        print(f"= Trainig_epoch          : {self.TRAINING_EPOCH}")
        print(f"= Weight_init            : {self.WEIGHT_INIT}")
        print(f"= Optimizer              : {self.OPTIMIZER}")
        print(f"= Weight_decay           : {self.WEIGHT_DECAY}")
        print(f"= Dropout                : {self.DROPOUT}")
