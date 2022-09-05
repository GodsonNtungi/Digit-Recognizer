import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow
from tensorflow import keras

from PIL import Image, ImageOps

# import model
model = keras.models.load_model('Model/digitrecognizer98.h5')
canvas_result = st_canvas(
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    update_streamlit=True,
    height=300,
    drawing_mode='freedraw',
    key="canvas",
)

if canvas_result.image_data is not None:
    # converting an image to 28,28 shape for the model
    image = Image.fromarray(np.array(canvas_result.image_data))
    image = image.resize((28, 28))
    # converting image to black and white
    image = ImageOps.grayscale(image)
    image = np.array(image)
    # input image to the model
    prediction = model.predict(image[None, :, :])
    error_threshold = 0.1
    # order the prediction output
    results = [[i, r] for i, r in enumerate(prediction[0]) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)

    for result in results:
        # displaying the result
        st.markdown(f'#### {result[0]} ')
        st.write(f'percent {round(result[1] * 100, 2)}%')
