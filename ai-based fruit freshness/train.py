import os
import json
import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import EfficientNetB0


# dataset location

DATASET_DIR = "Processed Dataset"


# image size

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16


# data augmentation

image_generator = ImageDataGenerator(

    rescale=1 / 255,

    validation_split=0.2,

    rotation_range=25,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.2,

    shear_range=0.2,

    horizontal_flip=True,

    brightness_range=[0.8, 1.2],

    fill_mode="nearest"

)


# training data

train_data = image_generator.flow_from_directory(

    DATASET_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True

)


# validation data

validation_data = image_generator.flow_from_directory(

    DATASET_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False

)


# save labels

class_names = list(

    train_data.class_indices.keys()

)

os.makedirs("model", exist_ok=True)

with open("model/labels.json", "w") as file:

    json.dump(

        class_names,

        file

    )


print("\nDetected Classes:\n")

print(class_names)


# EfficientNet model

base_model = EfficientNetB0(

    weights="imagenet",

    include_top=False,

    input_shape=(224, 224, 3)

)


# freeze layers first

base_model.trainable = False


# custom layers

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.3)(x)

x = Dense(

    256,

    activation="relu"

)(x)

x = Dropout(0.2)(x)

output = Dense(

    len(class_names),

    activation="softmax"

)(x)


# final model

model = Model(

    inputs=base_model.input,

    outputs=output

)


# compile model

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=0.0001

    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)


# callbacks

early_stop = EarlyStopping(

    monitor="val_accuracy",

    patience=5,

    restore_best_weights=True

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=2,

    verbose=1

)


# training phase 1

print("\nStarting Initial Training...\n")

history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=10,

    callbacks=[

        early_stop,

        reduce_lr

    ]

)


# unfreeze top layers

base_model.trainable = True

for layer in base_model.layers[:-40]:

    layer.trainable = False


# recompile

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=0.00001

    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)


# fine tuning

print("\nStarting Fine Tuning...\n")

history_fine = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=10,

    callbacks=[

        early_stop,

        reduce_lr

    ]

)


# save model

model.save(

    "model/fruit_model.h5"

)

print("\nTraining Completed Successfully")