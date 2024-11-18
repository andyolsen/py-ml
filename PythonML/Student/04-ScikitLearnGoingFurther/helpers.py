import matplotlib.pyplot as plt

def draw_vector(p0, p1):
    ax = plt.gca()
    ax.annotate('', p1, p0, 
                arrowprops={'arrowstyle': '->', 'linewidth': 4})
