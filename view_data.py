import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.widgets import Slider


def view_synthetic_data():
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.25)

    # Initial state
    init_patient = 7
    init_slice = 5

    def get_image(patient_id, slice_id):
        image_path = f"data/synthetic_baseline/{patient_id}_slice{slice_id}.npy"
        if os.path.exists(image_path):
            img_array = np.load(image_path)
            if img_array.ndim > 2:
                img_array = img_array.squeeze()
                if img_array.ndim > 2:
                    img_array = img_array[0]
            return img_array
        # Return a blank image if file is missing
        return np.zeros((224, 224))

    # Plot initial image
    img_data = get_image(init_patient, init_slice)
    img_plot = ax.imshow(img_data, cmap="gray")
    ax.set_title(f"Synthetic MRI - Patient {init_patient} (Slice {init_slice})")
    ax.axis("off")

    # Add interactive sliders
    ax_patient = plt.axes([0.2, 0.1, 0.6, 0.03])
    patient_slider = Slider(
        ax=ax_patient,
        label="Patient ID",
        valmin=1,
        valmax=30,
        valinit=init_patient,
        valstep=1,
    )

    ax_slice = plt.axes([0.2, 0.05, 0.6, 0.03])
    slice_slider = Slider(
        ax=ax_slice,
        label="Slice Depth",
        valmin=1,
        valmax=5,
        valinit=init_slice,
        valstep=1,
    )

    # Update function when sliders are moved
    def update(val):
        p_id = int(patient_slider.val)
        s_id = int(slice_slider.val)

        new_img = get_image(p_id, s_id)
        img_plot.set_data(new_img)

        # Adjust display contrast so it doesn't look washed out
        if new_img.max() > new_img.min():
            img_plot.set_clim(vmin=new_img.min(), vmax=new_img.max())

        ax.set_title(f"Synthetic MRI - Patient {p_id} (Slice {s_id})")
        fig.canvas.draw_idle()

    # Bind the sliders to the update function
    patient_slider.on_changed(update)
    slice_slider.on_changed(update)

    print("Opening Interactive Synthetic Data Viewer...")
    plt.show()


if __name__ == "__main__":
    view_synthetic_data()
