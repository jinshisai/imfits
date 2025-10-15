# image utils

# modules
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path



def get_contour_size(
    x: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    thr: float):
    '''
    Get contour size.

    Parameters
    ----------
    x, y (2D array): x and y coordinates of the data
    d (2D array): 2D image data
    thr (float): Threshold to draw contours
    '''

    # for debug, keep figure
    fig, ax = plt.subplots(1, 1)

    # get contour
    CS = ax.contour(x, y, d, levels=[thr], 
        colors = 'k')

    # get paths from contour
    paths = CS.get_paths()
    paths = paths[0]
    paths_splited = split_enclosed_contours(paths)
    # pick up largest area
    lengths = [len(p.vertices) for p in paths_splited]
    path = paths_splited[np.argmax(lengths)]

    # get (x,y) of the largest area
    v = path.vertices
    x = v[:,0]
    y = v[:,1]
    # to check
    ax.plot(x, y, color='r', lw=2., alpha=0.7)
    #plt.show() # for debug
    plt.close()

    # takeing mean
    x_mn = np.mean(x)
    y_mn = np.mean(y)
    r = np.sqrt((x - x_mn)**2 + (y - y_mn)**2)
    r_med = np.median(r)
    #r_sig = np.std(r)

    return r_med


def polygon_area(vertices):
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def split_enclosed_contours(path):
    """Split a matplotlib Path object into individual closed contours."""
    verts = path.vertices
    codes = path.codes

    if codes is None:
        # No codes: assume single open path
        return [Path(verts)]

    contours = []
    current_verts = []

    for vert, code in zip(verts, codes):
        if code == Path.MOVETO:
            # Start new contour
            if current_verts:
                contours.append(
                    Path(np.array(current_verts), codes=[Path.MOVETO] + [Path.LINETO]*(len(current_verts)-2) + [Path.CLOSEPOLY]))
            current_verts = [vert]
        elif code == Path.CLOSEPOLY:
            # Close current contour
            current_verts.append(vert)
            contours.append(
                Path(np.array(current_verts), codes=[Path.MOVETO] + [Path.LINETO]*(len(current_verts)-2) + [Path.CLOSEPOLY]))
            current_verts = []
        else:
            current_verts.append(vert)


    # force closing contour
    if code != Path.CLOSEPOLY:
        # Close current contour
        contours.append(
            Path(np.array(current_verts), codes=[Path.MOVETO] + [Path.LINETO]*(len(current_verts)-2) + [Path.CLOSEPOLY]))
        #current_verts = []

    return contours