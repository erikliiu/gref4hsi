"""
Minimal test to verify FuncAnimation works correctly
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numpy as np

# Create simple data
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)
(line,) = ax.plot([], [], "b-", linewidth=2)

# Animation state
anim_running = True
current_frame = 0

# Create play/pause button
button_ax = plt.axes([0.81, 0.05, 0.1, 0.04])
play_button = Button(button_ax, "Pause")


def update(frame):
    global current_frame
    current_frame = frame
    line.set_data(x[:frame], y[:frame])
    ax.set_title(f"Frame: {frame}")
    return (line,)


def animate(frame):
    global current_frame, anim_running
    if not anim_running:
        return update(current_frame)
    else:
        return update(frame)


anim = FuncAnimation(fig, animate, frames=len(x), interval=50, blit=False, repeat=True)


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

print("Starting test animation...")
print("If the line doesn't animate, FuncAnimation setup is broken")
plt.show()
