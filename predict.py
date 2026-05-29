import cv2
import json
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image


# load trained model

model = tf.keras.models.load_model(
    "model/fruit_model.h5"
)


# load labels

with open("model/labels.json", "r") as file:

    class_names = json.load(file)


# market price ranges

fruit_prices = {

    "Apple": (110, 140),

    "Banana": (50, 80),

    "Grape": (120, 180),

    "Guava": (60, 90),

    "Jujube": (150, 220),

    "Orange": (70, 110),

    "Pomegranate": (180, 260),

    "Strawberry": (250, 400)

}


nutrition_data = {

    "Apple": "Rich in Fiber and Vitamin C",

    "Banana": "High in Potassium and Energy",

    "Grape": "Contains antioxidants and vitamins",

    "Guava": "Excellent source of Vitamin C",

    "Jujube": "Rich in Iron and antioxidants",

    "Orange": "Boosts immunity naturally",

    "Pomegranate": "Improves blood circulation",

    "Strawberry": "Rich in antioxidants"

}


storage_tips = {

    "Apple": "Store inside refrigerator",

    "Banana": "Keep at room temperature",

    "Grape": "Refrigerate after purchase",

    "Guava": "Store in cool dry place",

    "Jujube": "Keep refrigerated",

    "Orange": "Avoid direct sunlight",

    "Pomegranate": "Refrigerate after cutting",

    "Strawberry": "Consume quickly after refrigeration"

}


def prepare_image(img_path):

    img = image.load_img(

        img_path,

        target_size=(224, 224)

    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(

        img_array,

        axis=0

    )

    return img_array


def detect_rotten_area(frame):

    hsv = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2HSV

    )

    dark_mask = cv2.inRange(

        hsv,

        np.array([0, 0, 0]),

        np.array([180, 255, 60])

    )

    rotten_pixels = cv2.countNonZero(
        dark_mask
    )

    total_pixels = (

        frame.shape[0] *

        frame.shape[1]

    )

    rotten_percentage = int(

        (rotten_pixels / total_pixels) * 100

    )

    heatmap = cv2.applyColorMap(

        dark_mask,

        cv2.COLORMAP_JET

    )

    overlay = cv2.addWeighted(

        frame,

        0.75,

        heatmap,

        0.25,

        0

    )

    cv2.imwrite(

        "temp/processed_fruit.jpg",

        overlay

    )

    return rotten_percentage


def freshness_check(frame):

    hsv = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2HSV

    )

    brightness = np.mean(hsv[:, :, 2])

    saturation = np.mean(hsv[:, :, 1])

    if brightness > 150 and saturation > 80:

        return "Fresh", "5 to 7 Days", 90

    elif brightness > 100:

        return "Medium Fresh", "2 to 4 Days", 65

    else:

        return "Rotten", "0 to 1 Day", 25


def quality_score(freshness, rotten):

    score = freshness / 10

    score -= rotten * 0.03

    score = max(1, min(score, 10))

    return round(score, 1)


def quality_grade(score):

    if score >= 8:

        return "Grade A"

    elif score >= 5:

        return "Grade B"

    return "Grade C"


def recommendation(condition):

    if condition == "Fresh":

        return "Premium quality fruit ready for sale"

    elif condition == "Medium Fresh":

        return "Sell quickly for best quality"

    return "Use for juice or processing"


def generate_price(fruit_name, condition):

    min_price, max_price = fruit_prices.get(

        fruit_name,

        (60, 120)

    )

    market_price = random.randint(

        min_price,

        max_price

    )

    if condition == "Fresh":

        selling_price = market_price

    elif condition == "Medium Fresh":

        selling_price = int(
            market_price * 0.7
        )

    else:

        selling_price = int(
            market_price * 0.4
        )

    return market_price, selling_price


def predict_fruit(img_path):

    prepared = prepare_image(
        img_path
    )

    prediction = model.predict(

        prepared,

        verbose=0

    )

    predicted_index = np.argmax(
        prediction
    )

    confidence = int(

        np.max(prediction) * 100

    )

    fruit_name = class_names[
        predicted_index
    ]

    frame = cv2.imread(img_path)

    frame = cv2.resize(

        frame,

        (700, 500)

    )

    rotten_percentage = detect_rotten_area(
        frame
    )

    condition, shelf_life, freshness = freshness_check(
        frame
    )

    score = quality_score(

        freshness,

        rotten_percentage

    )

    grade = quality_grade(score)

    advice = recommendation(condition)

    market_price, selling_price = generate_price(

        fruit_name,

        condition

    )

    return {

        "fruit": fruit_name,

        "condition": condition,

        "days": shelf_life,

        "freshness": freshness,

        "confidence": confidence,

        "rotten": rotten_percentage,

        "score": score,

        "grade": grade,

        "market_price": market_price,

        "selling_price": selling_price,

        "nutrition":
        nutrition_data.get(fruit_name),

        "storage":
        storage_tips.get(fruit_name),

        "recommendation":
        advice

    }