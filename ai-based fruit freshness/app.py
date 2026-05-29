import os
import sqlite3

from datetime import datetime

import pandas as pd
import streamlit as st

from PIL import Image

from predict import predict_fruit


# folders

os.makedirs("temp", exist_ok=True)


# streamlit config

st.set_page_config(

    page_title="FreshSense AI",

    page_icon="🧠",

    layout="wide"

)


# database

conn = sqlite3.connect(

    "fruit_history.db",

    check_same_thread=False

)

cursor = conn.cursor()

cursor.execute("""

CREATE TABLE IF NOT EXISTS scans (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    fruit TEXT,

    condition TEXT,

    price TEXT,

    scanned_time TEXT

)

""")

conn.commit()


# futuristic css

st.markdown("""

<style>

#MainMenu {
    visibility:hidden;
}

footer {
    visibility:hidden;
}

header {
    visibility:hidden;
}

.stApp {

    background:
    radial-gradient(
        circle at top,
        #111827,
        #020617,
        #000000
    );

    color:white;
}

.block-container {

    max-width:1450px;

    padding-top:1rem;
}


/* title */

.main-title {

    text-align:center;

    font-size:78px;

    font-weight:900;

    margin-bottom:5px;

    background:
    linear-gradient(
        to right,
        #00ffae,
        #3b82f6
    );

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;
}

.subtitle {

    text-align:center;

    color:#94a3b8;

    font-size:20px;

    margin-bottom:40px;
}


/* metrics */

[data-testid="stMetric"] {

    background:
    rgba(255,255,255,0.05);

    border:
    1px solid rgba(255,255,255,0.08);

    border-radius:22px;

    padding:18px;

    backdrop-filter:blur(12px);

    box-shadow:
    0px 0px 20px rgba(0,255,174,0.05);
}

[data-testid="stMetricLabel"] {

    color:#94a3b8;
}

[data-testid="stMetricValue"] {

    color:white;

    font-size:30px;
}


/* uploader */

section[data-testid="stFileUploader"] {

    background:
    rgba(255,255,255,0.04);

    border:
    1px solid rgba(255,255,255,0.08);

    padding:18px;

    border-radius:20px;
}


/* info boxes */

.stSuccess {

    border-radius:18px;
}

.stInfo {

    border-radius:18px;
}

.stWarning {

    border-radius:18px;
}


/* dataframe */

[data-testid="stDataFrame"] {

    border-radius:20px;

    overflow:hidden;
}


/* images */

img {

    border-radius:20px;
}


/* chart */

canvas {

    border-radius:20px;
}

</style>

""", unsafe_allow_html=True)


# header

st.markdown(

    """
    <div class="main-title">
    FreshSense AI
    </div>

    <div class="subtitle">
    AI Powered Fruit Freshness Intelligence Platform
    </div>
    """,

    unsafe_allow_html=True

)


# metrics row

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(

        "Total Scans",

        "128"

    )

with m2:

    st.metric(

        "Fresh Fruits",

        "103"

    )

with m3:

    st.metric(

        "Medium Fresh",

        "18"

    )

with m4:

    st.metric(

        "Rotten Fruits",

        "7"

    )


st.markdown("<br>", unsafe_allow_html=True)


# uploader

uploaded_file = st.file_uploader(

    "Upload Fruit Image",

    type=["jpg", "jpeg", "png"]

)


# prediction

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    with open("temp/temp.jpg", "wb") as file:

        file.write(uploaded_file.getbuffer())

    result = predict_fruit(
        "temp/temp.jpg"
    )


    # save scan history

    cursor.execute("""

    INSERT INTO scans (

        fruit,
        condition,
        price,
        scanned_time

    )

    VALUES (?, ?, ?, ?)

    """, (

        result["fruit"],

        result["condition"],

        f"₹{result['selling_price']}",

        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    ))

    conn.commit()


    # layout

    left, right = st.columns([1.2, 1])


    # images

    with left:

        img1, img2 = st.columns(2)

        with img1:

            st.subheader(
                "Original Image"
            )

            st.image(

                image,

                use_container_width=True

            )

        with img2:

            st.subheader(
                "Rotten Detection"
            )

            st.image(

                "temp/processed_fruit.jpg",

                use_container_width=True

            )


    # metrics

    with right:

        r1, r2 = st.columns(2)

        with r1:

            st.metric(

                "Fruit",

                result["fruit"]

            )

            st.metric(

                "Confidence",

                f"{result['confidence']}%"

            )

            st.metric(

                "Quality Score",

                result["score"]

            )

            st.metric(

                "Market Price",

                f"₹{result['market_price']}/kg"

            )

            st.metric(

                "Rotten Area",

                f"{result['rotten']}%"

            )

        with r2:

            st.metric(

                "Condition",

                result["condition"]

            )

            st.metric(

                "Shelf Life",

                result["days"]

            )

            st.metric(

                "Grade",

                result["grade"]

            )

            st.metric(

                "Selling Price",

                f"₹{result['selling_price']}/kg"

            )


        st.success(

            f"🤖 {result['recommendation']}"

        )

        st.info(

            f"🥗 {result['nutrition']}"

        )

        st.warning(

            f"💡 {result['storage']}"

        )


    st.markdown("<br>", unsafe_allow_html=True)


    # analytics

    st.subheader(
        "Recent Scan History"
    )

    history = pd.read_sql_query(

        "SELECT * FROM scans ORDER BY id DESC LIMIT 5",

        conn

    )

    st.dataframe(

        history,

        use_container_width=True

    )