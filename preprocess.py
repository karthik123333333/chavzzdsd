import os
import cv2
import numpy as np

from tqdm import tqdm


# dataset folders

DATASET_DIR = "Fruit Freshness Dataset"

OUTPUT_DIR = "Processed Dataset"


# create output folder

os.makedirs(

    OUTPUT_DIR,

    exist_ok=True

)


# image size

IMAGE_SIZE = (224, 224)


# preprocessing function

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:

        return None

    
    # resize

    image = cv2.resize(

        image,

        IMAGE_SIZE

    )


    # convert to LAB color space

    lab = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2LAB

    )


    # split channels

    l, a, b = cv2.split(lab)


    # CLAHE for lighting improvement

    clahe = cv2.createCLAHE(

        clipLimit=3.0,

        tileGridSize=(8, 8)

    )

    l = clahe.apply(l)


    # merge channels

    merged = cv2.merge((l, a, b))


    # convert back

    image = cv2.cvtColor(

        merged,

        cv2.COLOR_LAB2BGR

    )


    # gaussian blur

    image = cv2.GaussianBlur(

        image,

        (3, 3),

        0

    )


    # normalize

    image = image.astype("float32") / 255.0


    return image


# process all folders

for fruit_name in os.listdir(DATASET_DIR):

    fruit_path = os.path.join(

        DATASET_DIR,

        fruit_name

    )

    output_fruit_path = os.path.join(

        OUTPUT_DIR,

        fruit_name

    )

    os.makedirs(

        output_fruit_path,

        exist_ok=True

    )

    print(f"\nProcessing {fruit_name} images...\n")

    for image_name in tqdm(

        os.listdir(fruit_path)

    ):

        image_path = os.path.join(

            fruit_path,

            image_name

        )

        processed = preprocess_image(

            image_path

        )

        if processed is not None:

            save_path = os.path.join(

                output_fruit_path,

                image_name

            )

            processed = (

                processed * 255

            ).astype("uint8")

            cv2.imwrite(

                save_path,

                processed

            )

print("\nDataset preprocessing completed")