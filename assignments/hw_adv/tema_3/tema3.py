import math
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import clear_output, display

# close import * leaks
__all__ = [
    "FG_POINT", "BG_POINT", "U_3000",
    "draw_square", "draw_right_triangle", "draw_inverted_pyramid", "draw_rhombus",
    "draw_circle", "draw_circle_shape",
    "draw_plot", "draw_plot_rotating",
]

FG_POINT = '🔘' # ⚪️ 🔘 ⚫️ 🍔
BG_POINT = '⚫️' # ⬤ ◯ • · 
U_3000 = '　'

plt.style.use('dark_background')

def draw_square(height):
    for _ in range(height):
        print('* ' * height)


def draw_right_triangle(height):
    for i in range(height + 1):
        print('* ' * i)


def draw_inverted_pyramid(height):
    for i in range(height, 0, -1):
        print(f"{' ' * (height - i)}{'* ' * i}")


def draw_rhombus(height):
    height = height + height % 2 # force even up one
    height = max(4, height) # force min height

    for i in range(0, height):
        spaces = height - i if i < height // 2 else i
        stars = i if i < height // 2 else height - i
        print(" " * spaces, end='')
        print("* " * stars, end='')
        print()


def get_circle_points(dots: int = 360, *, offset_deg: float = 0.0, radius: float = 1.0) -> list[dict[float, float]]:
    if dots < 1:
        raise ValueError("dots must be >= 1")

    circle_points = []
    offset_rad = math.radians(offset_deg) #  # convert deg rotation to radians

    for i in range(dots):
        theta = 2 * math.pi * i / dots + offset_rad # convert dots to radians + rotation
        x = math.cos(theta) * radius
        y = math.sin(theta) * radius
        circle_points.append({'x': x, 'y': y})

    return circle_points


def initialize_graph(size=10):
    graph = []
    for _ in range(size):
        graph.append([BG_POINT for _ in range(size)])
    return graph


def remap_range(value: float, range_initial: tuple, range_new: tuple) -> float:
    """ desc made by gpt\n
    Takes any range and remaps it, works both ways with positive and negative numbers

    Args:
        value (float): Input value.
        range_initial (tuple[float, float]): (a, b) source interval.
        range_new (tuple[float, float]): (lo, hi) target interval.
    Returns:
        float: Remapped value.
    Raises:
        ValueError: If source has zero length or value is outside it.
    """

    lo_ini, hi_ini = range_initial
    lo_new, hi_new = range_new

    size_ini = hi_ini - lo_ini
    size_new = hi_new - lo_new

    if lo_ini > value > hi_ini:
        raise ValueError(f"value: [{value}], outside of range: [{range_initial}]")
    if size_ini == 0:
        raise ValueError("range size cannot be 0")

    distance_from_back = hi_ini - value
    perc = lo_new + (distance_from_back / size_ini) * size_new
    # perc100 = distance_from_back * 100 / range_size

    return perc


def dumb_fill(graph):
    for row in graph:
        try:
            left = row.index(FG_POINT)
            right = len(row) - row[::-1].index(FG_POINT)
            for i in range(left, right - 1):
                row[i] = FG_POINT
        except ValueError:
            continue



def bresenham(p0, p1): # draw pixel by pixel
    # integer dumb line
    x0, y0 = round(p0[0]), round(p0[1])
    x1, y1 = round(p1[0]), round(p1[1]) 
    dx, dy = abs(x1-x0), abs(y1-y0) # get distance (horizontal and vertical)
    sx = 1 if x0 < x1 else -1 # choose direction x
    sy = 1 if y0 < y1 else -1 # choose direction y
    err = dx - dy # signed decision error
    pts = []
    while True:
        pts.append({'x': x0, 'y': y0}) # add new point
        if x0 == x1 and y0 == y1: return pts # finished drawing
        e2 = 2*err # scale error for cheap updates
        if e2 > -dy: # step vertical
            err -= dy; # decrement err by abs distance x
            x0 += sx # shift x to target x by 1
        if e2 < dx: # step horizontal
            err += dx; # decrement err by abs distance y
            y0 += sy # shift y to target y by 1


def draw_line_slope(graph): # not implemented change to points (-1.0 <-> 1.0) instead of graph idiot
    p = []

    for x, row in enumerate(graph):
        for y, cell in enumerate(row):
            if cell == FG_POINT:
                p.append((x, y))

    p.append(p[0])

    for i in range(len(p) - 1):
        x1, y1 = p[i][0]  , p[i][1]
        x2, y2 = p[i+1][0], p[i+1][1]
        print(x1, y1, x2, y2)

        if x2 - x1 == 0:
            m = 0
            b = 0
        else:
            m = (y2 - y1) / (x2 - x1) # slope
            b = y1 - m * x1 # intercept

        print(f"y = {m}x + {b}")


def draw_on_graph_plain(graph_p: list[list], points: list[dict]):
        for point in points:
            new_x = point['x']
            new_y = point['y']

            graph_p[round(new_y)][round(new_x)] = FG_POINT


def draw_on_graph_rescale(graph: list[list], points: list[dict], graph_size: int, r=1.0):
    for point in points:
        new_x = remap_range(point['x'], (-r, r), (0, graph_size - 1))
        new_y = remap_range(point['y'], (-r, r), (0, graph_size - 1))

        graph[round(new_y)][round(new_x)] = FG_POINT


def draw_circle(height = 17, *, fill: bool = False, shape: int = 360):
    graph = initialize_graph(height)
    points = get_circle_points(shape)
    draw_on_graph_rescale(graph, points, height)
    if fill:
        dumb_fill(graph)
    for i in range(height):
        for y in range(height):
            print(graph[i][y], end='')
        print()


def draw_circle_shape(height = 17, *, fill: bool = False, shape: int = 4, rot_deg: float = 0.0):
    graph_p = initialize_graph(height)
    points = get_circle_points(shape, offset_deg=rot_deg)
    draw_on_graph_rescale(graph_p, points, height)

    line_points = points[:] # copy
    line_points.append(line_points[0]) # close shape

    for i in range(len(line_points) - 1): # draw line
        new_x0 = remap_range(line_points[i]['x'], (-1.0, 1.0), (0, height - 1))
        new_y0 = remap_range(line_points[i]['y'], (-1.0, 1.0), (0, height - 1))
        p0 = [new_x0, new_y0]

        new_x1 = remap_range(line_points[i + 1]['x'], (-1.0, 1.0), (0, height - 1))
        new_y1 = remap_range(line_points[i + 1]['y'], (-1.0, 1.0), (0, height - 1))
        p1 = [new_x1, new_y1]

        draw_on_graph_plain(graph_p, bresenham(p0, p1))

    if fill:
        dumb_fill(graph_p)

    for i in range(height):
        for y in range(height):
            print(graph_p[i][y], end='')
        print()


def draw_plot(shape = 6, rotation_degrees = 0, fill="line"):
    # chatgpt
    BG   = "#1e1e1e"
    EDGE = "#3794ff"
    FILL = "#21b3f6"

    pts = get_circle_points(shape, radius=1.0, offset_deg=rotation_degrees)

    xs, ys = zip(*((p['x'], p['y']) for p in pts))
    fig, ax = plt.subplots(figsize=(4,4), facecolor=BG)
    ax.set_facecolor(BG)

    xs = list(xs) + [xs[0]] # complete shape
    ys = list(ys) + [ys[0]]

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    if fill == "line":  # one smooth path
        ax.plot(xs, ys, linewidth=2, color=EDGE, antialiased=True,
                solid_joinstyle='round', solid_capstyle='round')

    elif fill == "fill":  # filled shape
        ax.fill(xs[:-1], ys[:-1],
                facecolor=FILL, edgecolor=EDGE,
                linewidth=2, alpha=0.3,  # lower alpha = easier on eyes
                joinstyle='round', capstyle='round', antialiased=True)

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    plt.margins(0) # remove extra padding
    plt.tight_layout(pad=0) # flush to edges
    plt.show()


def draw_plot_rotating(shape=6, fill="line", speed_deg=2, frames=360):
    # rotating version that replaces the plot in-place (VS Code/Jupyter)
    # chatgpt  
    BG="#1e1e1e"; EDGE="#3794ff"; FILL="#21b3f6"

    fig, ax = plt.subplots(figsize=(4,4), facecolor=BG)
    ax.set_facecolor(BG); ax.set_aspect('equal', adjustable='box'); ax.axis('off')
    plt.margins(0); plt.tight_layout(pad=0)

    # init
    pts0 = get_circle_points(shape, radius=1.0, offset_deg=0)
    xs0, ys0 = zip(*((p['x'], p['y']) for p in pts0))

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    if fill == "line":
        (line,) = ax.plot(list(xs0)+[xs0[0]], list(ys0)+[ys0[0]],
                          linewidth=2, color=EDGE, antialiased=True,
                          solid_joinstyle='round', solid_capstyle='round')
    else:
        poly = mpatches.Polygon(list(zip(xs0, ys0)), closed=True,
                                facecolor=FILL, edgecolor=EDGE,
                                linewidth=2, alpha=0.3, joinstyle='round', antialiased=True)
        ax.add_patch(poly)

    for deg in range(0, frames, speed_deg):
        pts = get_circle_points(shape, radius=1.0, offset_deg=deg)
        xs, ys = zip(*((p['x'], p['y']) for p in pts))
        if fill == "line":
            line.set_data(list(xs)+[xs[0]], list(ys)+[ys[0]])
        else:
            poly.set_xy(list(zip(xs, ys)))
        clear_output(wait=True); display(fig); time.sleep(0.02)

    plt.close(fig)


# console
# draw_square(5)
# draw_right_triangle(5)
# draw_inverted_pyramid(5)
# draw_rhombus(3)
# draw_circle(17, fill=True)
# draw_circle_shape(28, shape=5, fill=False) # pentagon
# draw_circle_shape(21, shape=6, fill=False) # hexagon


# matplotlib OLD
# pts = get_circle_points(360, radius=1.0)
# xs, ys = zip(*((p['x'], p['y']) for p in pts)) # some gpt voodoo
# plt.scatter(xs, ys, s=10)

# pts = get_circle_points(6, 1.0)
# xs, ys = zip(*((p['x'], p['y']) for p in pts))

# xs = list(xs) + [xs[0]] # janky complete
# ys = list(ys) + [ys[0]]

# fig, ax = plt.subplots(figsize=(4,4))
# ax.plot(xs, ys, lw=2, antialiased=True)     # one smooth path

# # plt.scatter(get_circle_points(360))
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# ax.set_xlim(-1.05, 1.05)
# ax.set_ylim(-1.05, 1.05)
# ax.axis('off')
# plt.show()