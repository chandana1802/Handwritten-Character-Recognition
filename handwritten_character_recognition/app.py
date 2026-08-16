import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# Load trained model
try:
    model = tf.keras.models.load_model("mnist_cnn.keras")
except Exception:
    messagebox.showerror(
        "Error",
        "Model not found!\n\n"
        "Please run train_model.py first."
    )
    raise SystemExit


# Window
window = tk.Tk()

window.title("Handwritten Digit Recognition")
window.geometry("500x650")
window.resizable(False, False)


# Title
title = tk.Label(
    window,
    text="Handwritten Digit Recognition",
    font=("Arial", 22, "bold")
)

title.pack(pady=15)


# Canvas
canvas = tk.Canvas(
    window,
    width=280,
    height=280,
    bg="black",
    cursor="cross"
)

canvas.pack(pady=10)


# Create image for drawing
image = Image.new(
    "L",
    (280, 280),
    color=0
)

draw = ImageDraw.Draw(image)


# Drawing function
def draw_digit(event):
    x = event.x
    y = event.y

    brush_size = 20

    canvas.create_oval(
        x - brush_size,
        y - brush_size,
        x + brush_size,
        y + brush_size,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        [
            x - brush_size,
            y - brush_size,
            x + brush_size,
            y + brush_size
        ],
        fill=255
    )


# Predict function
def predict_digit():

    # Resize to MNIST size
    img = image.resize((28, 28))

    # Convert to numpy array
    img_array = np.array(img)

    # Normalize
    img_array = img_array.astype("float32") / 255.0

    # Add dimensions
    img_array = img_array.reshape(
        1,
        28,
        28,
        1
    )

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    # Get predicted digit
    digit = np.argmax(prediction)

    # Confidence
    confidence = np.max(prediction) * 100

    # Display result
    result_label.config(
        text=f"Predicted Digit: {digit}\n"
             f"Confidence: {confidence:.2f}%"
    )


# Clear canvas
def clear_canvas():

    canvas.delete("all")

    draw.rectangle(
        [0, 0, 280, 280],
        fill=0
    )

    result_label.config(
        text="Draw a digit and click Predict"
    )


# Bind mouse movement
canvas.bind(
    "<B1-Motion>",
    draw_digit
)


# Predict button
predict_button = tk.Button(
    window,
    text="Predict",
    font=("Arial", 14, "bold"),
    command=predict_digit,
    width=15
)

predict_button.pack(pady=10)


# Clear button
clear_button = tk.Button(
    window,
    text="Clear",
    font=("Arial", 14),
    command=clear_canvas,
    width=15
)

clear_button.pack(pady=5)


# Result
result_label = tk.Label(
    window,
    text="Draw a digit and click Predict",
    font=("Arial", 16, "bold")
)

result_label.pack(pady=20)


# Start application
window.mainloop()