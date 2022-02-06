import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw_risk(data):    
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.plot([0, 4],[-0.2, 1], alpha=0)
    for i, l in enumerate(data):
        ax.add_patch(Rectangle((i, 0), 1, 1, facecolor=data[l], fill=True))
        ax.text(i + 0.01, -0.15, l)
    ax.set_aspect("equal")
    plt.show()
    
    