MINC2500_PATH = "minc-2500/images"
FMD_PATH = "fmd/image"
DATA_ROOT_PATH = "C:/Files/material_datasets"

MINC2500_INPUT_SHAPE = (224, 224, 3)
MINC2500_BATCH_SIZE = 25
EPOCHS_COUNT = 20

MINC2500_TRAIN_AMOUNT = 1500
MINC2500_VALIDATION_AMOUNT = 500
MINC2500_TEST_AMOUNT = 500

FMD_TRAIN_AMOUNT = 60
FMD_VALIDATION_AMOUNT = 20
FMD_TEST_AMOUNT = 20

CLASSES = [
            "fabric",
            "foliage",
            "glass",
            "leather",
            "metal",
            "paper",
            "plastic",
            "stone",
            "water",
            "wood"
        ]