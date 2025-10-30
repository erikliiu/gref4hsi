"""
Test the exact animation structure from MotionAnalyzer
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create test data
n_frames = 200
t = np.linspace(0, 4 * np.pi, n_frames)
x = np.cos(t)
y = np.sin(t)
z = t / (4 * np.pi)

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([0, 1])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Initialize line
(line,) = ax.plot([], [], [], "b-", linewidth=2, label="Trajectory")
(point,) = ax.plot([], [], [], "ro", markersize=8)

# Create slider
slider_ax = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor="lightgoldenrodyellow")
slider = Slider(slider_ax, "Frame", 0, n_frames - 1, valinit=0, valstep=1)

# Create button
button_ax = plt.axes([0.87, 0.02, 0.1, 0.04])
play_button = Button(button_ax, "Pause")

# Animation state
anim_running = True
current_frame = 0


# Update function
def update(frame):
    global current_frame
    current_frame = frame

    # Update line
    line.set_data(x[: frame + 1], y[: frame + 1])
    line.set_3d_properties(z[: frame + 1])

    # Update point
    if frame > 0:
        point.set_data([x[frame]], [y[frame]])
        point.set_3d_properties([z[frame]])

    # Update slider without triggering callback
    slider.eventson = False
    slider.set_val(frame)
    slider.eventson = True

    ax.set_title(f"Frame: {frame}/{n_frames}")

    return line, point


# Animation function
def animate(frame):
    global current_frame, anim_running
    if not anim_running:
        return update(current_frame)
    else:
        current_frame = frame
        return update(frame)


# Create animation
print("Creating FuncAnimation...")
anim = FuncAnimation(
    fig,
    animate,
    frames=n_frames,
    interval=50,
    blit=False,
    repeat=True,
    cache_frame_data=False,
)


# Slider callback
def slider_update(val):
    global current_frame, anim_running
    frame = int(slider.val)
    current_frame = frame
    # Pause when slider moved
    if anim_running:
        print("Slider moved - pausing")
        anim.event_source.stop()
        anim_running = False
        play_button.label.set_text("Play")
    update(frame)
    fig.canvas.draw_idle()


slider.on_changed(slider_update)


# Button callback
def toggle_animation(event):
    global anim_running
    if anim_running:
        print("Pausing animation")
        anim.event_source.stop()
        anim_running = False
        play_button.label.set_text("Play")
    else:
        print("Resuming animation")
        anim_running = True
        play_button.label.set_text("Pause")
        anim.event_source.start()
    fig.canvas.draw_idle()


play_button.on_clicked(toggle_animation)

print("Animation setup complete")
print("- Animation should start automatically")
print("- Use slider to jump to any frame (auto-pauses)")
print("- Use Play/Pause button to control animation")
plt.show()
