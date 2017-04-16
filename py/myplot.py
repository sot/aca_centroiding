import matplotlib.pylab as plt
import numpy as np
import Ska.Numpy


Bgd_Class_Names = ['FlightBgd', 'DynamBgd_Median', 'DynamBgd_SigmaClip']


def plot_d_ang(slot, slot_ref, key, dt, table, nstart=None, nstop=None):
    """
    Convert this into a chandra_aca tool
    """
    
    ok1 = table['slot'] == slot
    ok2 = table['slot'] == slot_ref

    fig = plt.figure(figsize=(7.5, 2.5))

    for i, bgd_class_name in enumerate(Bgd_Class_Names):
        if i==0:
            ax1 = plt.subplot(1, 3, 1)
            plt.ylabel('delta {} (arcsec)'.format(key))
        else:
            ax = plt.subplot(1, 3, i + 1, sharey=ax1)
            plt.setp(ax.get_yticklabels(), visible=False)
        ok = table['bgd_class_name'] == bgd_class_name
        ang_interp = Ska.Numpy.interpolate(table[ok * ok1][key][0],
                                           table[ok * ok1]['time'][0] + dt,
                                           table[ok * ok2]['time'][0],
                                           method="nearest")
        d_ang = table[ok * ok2][key][0] - ang_interp
        time = table[ok * ok2]['time'][0] - table[ok * ok2]['time'][0][0]
        
        if nstop is None:
            nstop = len(d_ang)
            
        if nstart is None:
            nstart = 0
            
        residuals = np.std(d_ang[nstart:nstop] - np.median(d_ang[nstart:nstop]))
        
        plt.plot(time, d_ang, color='Darkorange',
                      label='std = {:.3f}'.format(residuals))
        plt.xlabel('Time (sec)')
        plt.title(bgd_class_name)
        plt.legend(loc='best')
        plt.margins(0.05)
        #if ylim is None:
        #    ylim = plt.gca().get_ylim()
        #else:
        #    plt.ylim(ylim)            

    plt.subplots_adjust(left=0.05, right=0.99,
                         bottom=0.2, top=0.9,
                         hspace=0.3, wspace=0.1)
    return


def plot_px_history(table, keys, hot_pixels=None, slot=0, mag=None,
                    bgd_class_name='FlightBgd',
                    legend_text="", title_text='Hot'):
    """
    Plot time series of a given ACA pixel value.

    :param table: table with results of call to centroids() (see centroids.py)
    :param keys: list containing pixel coordinates,
                 e.g. [(120, 210), (30, 150)]
    :param slot: slot number
    :param mag: magnitude
    :param bgd_class_name: class used for background calculation (see classes.py)
    """

    n = len(keys)
    ll = 3 * n
    fig = plt.figure(figsize=(6, ll))

    ok1 = table['slot'] == slot
    ok2 = table['bgd_class_name'] == bgd_class_name
    ok = ok1 * ok2
    if mag is not None:
        ok3 = table['mag'] == mag
        ok = ok * ok3

    for i, key in enumerate(keys):

        plt.subplot(n, 1, i + 1)
        deques = table[ok]['deque_dict'][0]
        time = table[ok]['time'][0] - table[ok]['time'][0][0]

        px_vals = []
        bgd_vals = []

        current_bgd_val = 0

        for i, deque in enumerate(deques):
            if key in deque.keys():
                current_px_val = deque[key][-1]
                if bgd_class_name == 'DynamBgd_Median':
                    current_bgd_val = np.median(deque[key])
                elif bgd_class_name == 'DynamBgd_SigmaClip':
                    current_bgd_val = sigma_clip(deque[key])
                px_vals.append(current_px_val * 5./1.7) # e-/sec
                bgd_vals.append(current_bgd_val * 5./1.7)
            else:
                px_vals.append(-1)
                bgd_vals.append(-1)

        if legend_text == 'Simulated':
            plt.plot(table['time'][0] - table['time'][0][0], hot_pixels[key],
                     label=legend_text, color='gray')
        plt.plot(time, px_vals, label='Sampled', color='slateblue')
        plt.plot(time, bgd_vals, label="Derived", color='darkorange', lw=2)
        plt.xlabel('Time (sec)')
        plt.ylabel('Pixel value (e-/sec)')
        plt.title('{} pixel coordinates = {}'.format(title_text, key))
        plt.legend()
        plt.grid()
        plt.margins(0.05)

    plt.subplots_adjust(left=0.05, right=0.99,
                        top=0.9, bottom=0.2,
                        hspace=0.5, wspace=0.3)
    return


def plot_coords_excess(slot, table, coord):
    """
    Plot difference between true centroid coordinate (row or col or yan or zan)
    and derived centroid coordinates, as a function of time.

    Compute and display standard deviation of residuals.

    :param slot: slot number
    :param table: table with results of call to centroids() (see centroids.py)
    :param coord: name of coordinate, one of 'row', 'col', 'yan', 'zan'
    """

    fig = plt.figure(figsize=(9, 6))
    color = ['green', 'red', 'blue']

    for i, bgd_class_name in enumerate(Bgd_Class_Names):
        ok = table['bgd_class_name'] == bgd_class_name
        excess = table[ok][coord][slot] - table[ok]["true_" + coord][slot]
        std = np.std(excess - np.median(excess))
        time = table['time'][slot] - table['time'][slot][0]
        plt.plot(time, excess, color=color[i],
                 label="std = {:.3f}, ".format(std) + bgd_class_name)
        plt.margins(0.05)

    if coord in ['yan', 'zan']:
        unit = '(arcsec)'
    else:
        unit = '(pixel)'
    plt.ylabel(coord + " - true " + coord + " " + unit)
    plt.xlabel("Time (sec)")
    plt.title("Difference between derived " + coord +
              " and true " + coord + " coordinates");
    plt.grid()
    plt.legend()
    return


def plot_coords(slot, table, coord, mag=None):
    """
    Plot row or col or yan or zan centroid as a function of time.
    Add corresponding 'true' centroid in case of simulated data.

    :param slot: slot number
    :param table: table with results of call to centroids() (see centroids.py)
    :param coord: name of coordinate, one of 'row', 'col', 'yan', 'zan'
    """

    fig = plt.figure(figsize=(9, 6))
    color = ['green', 'red', 'blue']

    for i, bgd_class_name in enumerate(Bgd_Class_Names):
        #print '{:.2f}'.format(np.median(t[coord][slot]))
        ok1 = table['bgd_class_name'] == bgd_class_name
        ok2 = table['slot'] == slot
        ok = ok1 * ok2
        if mag is not None:
            ok3 = table['mag'] == mag
            ok = ok * ok3
        time = table[ok]['time'][0] - table[ok]['time'][0][0]
        plt.plot(time, table[ok][coord][0], color=color[i], label=bgd_class_name)
        plt.margins(0.05)

    text = ""
    if "true_" + coord in table.colnames:
        time = table['time'][slot] - table['time'][slot][0]
        plt.plot(time, table[ok]["true_" + coord][slot],
                 '--', color='k', lw='2', label="True")
        text = " and true " + coord + " coordinates."
    plt.ylabel(coord)
    plt.xlabel("Time (sec)")
    plt.title("Derived " + coord + text);
    plt.grid()
    plt.legend()
    return


def plot_coords_ratio(table1, table2, coord, slot=0, mag=None,
                      bgd_class_name='FlightBgd'):
    """
    Plot ratio of centroid coordinate (row or col or yan or zan) derived using
    different number of samples (ndeque).

    :param table1: table with results of call to centroids() (see centroids.py)
    :param table2: table with results of call to centroids() (see centroids.py)
    :param slot: slot number
    :param coord: name of coordinate, one of 'row', 'col', 'yan', 'zan'
    :param mag: magnitude
    :param bgd_class_name: algorithm for background calculation

    """

    coords = []

    for i, tab in enumerate([table1, table2]):
        ok1 = tab['bgd_class_name'] == bgd_class_name
        ok2 = tab['slot'] == slot
        ok = ok1 * ok2
        if mag is not None:
            ok3 = tab['mag'] == mag
            ok = ok * ok3
        coords.append(tab[ok][coord][0])

    time = tab[ok]['time'][0] - tab[ok]['time'][0][0]
    plt.plot(time, coords[0]/coords[1], color='darkorange', label=bgd_class_name)
    plt.xlabel("Time (sec)")
    plt.ylabel("{} coord ratio".format(coord))
    plt.grid()
    plt.legend()
    plt.margins(0.05)
    return


def plot_star_image(data):
    """
    Plot star image
    """

    c32 = np.array([1, 2, 2, 6, 6, 7, 7, 6, 6, 2, 2, 1, 1]) - 0.5
    r32 = np.array([2, 2, 1, 1, 2, 2, 6, 6, 7, 7, 6, 6, 2]) - 0.5

    c6x6 = [0.5, 6.5, 6.5, 0.5, 0.5]
    r6x6 = [0.5, 0.5, 6.5, 6.5, 0.5]

    plt.imshow(data, cmap=plt.get_cmap('jet'), interpolation='none',
               origin='lower')
    #plt.colorbar()
    plt.plot(c32, r32, '--', lw=2, color='w')
    plt.plot(c6x6, r6x6 , '-', lw=2, color='w')
    plt.xlim(-0.5, 7.5)
    plt.ylim(-0.5, 7.5)
    return


def patch_coords(row0, col0, img_size):

    r_min = np.int(row0.min())
    r_max = np.int(row0.max() + img_size)
    c_min = np.int(col0.min())
    c_max = np.int(col0.max() + img_size)

    return [r_min, r_max, c_min, c_max]


def plot_image(img, img_number, row0, col0, img_size, vmin=0, vmax=600):
    """
    Plot 6x6 or 8x8 background image

    :param img: current bgd image, always 8x8
    :param img_number: number of image, or number of time frame,
                       to fetch appropriate item from row0/col0
    :param row0: list with size=nframes containing IMGROW0
    :param row0: list with size=nframes containing IMGCOL0
    :img_size: image size, 8px for ER, 6px for Science,
               to fetch and plot bgd image part used for bgd subtraction
    """

    r_min, r_max, c_min, c_max = patch_coords(row0, col0, img_size)

    # +3 chosen arbitrary, to plot a bit larger area
    data = vmin * np.ones((c_max - c_min + 3, r_max - r_min + 3))

    dr = row0[img_number] - r_min
    dc = col0[img_number] - c_min

    data[dr:dr + img_size, dc:dc + img_size] = img[:img_size, :img_size]

    im = plt.imshow(data, cmap=plt.get_cmap('hot'), interpolation='none',
                    origin='lower', vmin=vmin, vmax=vmax)

    return im


def plot_images(table, n_start, n_stop, slot=0, mag=None, img_size=8,
                bgd_class_name='FlightBgd', vmin=0, vmax=600, colname="bgdimg"):
    """
    Plot a collection of background images, between img number
    n_start and n_stop.
    """
    
    if colname not in ['img', 'imgraw', 'bgdimg']:
        raise ValueError("plot_images:: "
                         "Text expected to be in ['img', 'imgraw', 'bgdimg']")

    ok1 = table['bgd_class_name'] == bgd_class_name
    ok2 = table['slot'] == slot
    ok = ok1 * ok2
    if mag is not None:
        ok3 = table['mag'] == mag
        ok = ok * ok3

    times = table['time'][0]
    delta_t = times[1] - times[0]

    fig = plt.figure(figsize=(8, 25))
    
    data = []
    for i, aa in enumerate(table[ok][colname][0][n_start:n_stop]):
        #plt.subplot(12, 10, i + 1)
        plt.subplot(12, 7, i + 1)
        row0 = table[ok]['row0'][0]
        col0 = table[ok]['col0'][0]
        index = i + n_start
        aa = aa.reshape(8, 8) # imgraw 64, bgdimg 8x8
        im = plot_image(aa, index, row0, col0, img_size,
                            vmin=vmin, vmax=vmax)
        #plt.title('t{:4.0f}:\n{}, {}'.format(index * delta_t, row0[index], col0[index]));
        plt.title('{:4.0f}:\n{}, {}'.format(index, row0[index], col0[index]));
        plt.axis('off')
        data.append(aa)

    plt.subplots_adjust(left=0.0, right=0.9,
                        top=0.9, bottom=0.2,
                        wspace=0.2, hspace=0.01)
    
    cbar_ax = fig.add_axes([0.92, 0.85, 0.01, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0, np.int(vmax/2), vmax])
    cbar.set_ticklabels(['0', '{:0d}'.format(np.int(vmax/2)), '{}'.format(vmax)])
    
    print("Format of the titles is 'frame #: imgrow0, imgcol0'")
    print("Plot {} from {} to {}".format(colname, n_start, n_stop))
    print("Bgd Class: {}, ndeque = {}".format(bgd_class_name, table[ok]['ndeque'][0]))

    return data


def plot_bgd_patch(deque_dict, img_number, row0, col0, img_size, bgd_class_name,
                   vmin=0, vmax=600):
    """
    Plot values of all bgd pixels that have been sampled so far.

    :param deque_dict: dictionary with keys being pixel coords,
                       e.g. (120, 210), and vals being lists of up to
                       ndeque pixel values
    :param img_number: number of image, or number of time frame,
                       needed to fetch appropriate item from row0/col0
    :param row0: list with size=nframes containing IMGROW0
    :param row0: list with size=nframes containing IMGCOL0
    :param img_size: image size, 8px for ER, 6px for Science,
                     to fetch and plot 6x6 bgd image section used for
                     bgd subtraction
    :param bgd_class_name: class used to compute background
    """
    
    if bgd_class_name not in ['DynamBgd_Median', 'DynamBgd_SigmaClip']:
        raise TypeError("plot_bgd_patch:: "
                        "bgd_class_name {} not supported" + bgd_class_name)

    r_min, r_max, c_min, c_max = patch_coords(row0, col0, img_size)

    data = vmin * np.ones((c_max - c_min + 3, r_max - r_min + 3))

    dr = np.int(row0[img_number] - r_min)
    dc = np.int(col0[img_number] - c_min)

    keys = []

    for rr in range(r_min, r_max + 1):
        for cc in range(c_min, c_max + 1):
            keys.append((rr, cc))

    for key in deque_dict.keys():
        if key in keys:
            if bgd_class_name == 'DynamBgd_Median':
                val = np.median(deque_dict[key])
            elif bgd_class_name == 'DynamBgd_SigmaClip':
                val = sigma_clip(deque_dict[key])
            data[np.int(key[0] - r_min), np.int(key[1] - c_min)] = val

    im = plt.imshow(data, cmap=plt.get_cmap('hot'), interpolation='none',
               origin='lower', vmin=vmin, vmax=vmax)

    return data, im


def sigma_clip(deque):
    """
    Compute sigma clipping of a list
    """
    if len(deque) > 2:
        d_min = np.argmin(deque)
        d_max = np.argmax(deque)
        m = np.zeros(len(deque))
        m[d_min] = 1
        m[d_max] = 1
        deque = np.ma.array(deque, mask=m)
    return np.mean(deque)


def plot_bgd_patches(table, n_start, n_stop, slot=0, mag=None, img_size=8,
                     bgd_class_name='DynamBgd_Median', vmin=0, vmax=600):
    """
    Plot a collection of background patches, between img number
    n_start and n_stop.
    """

    ok1 = table['bgd_class_name'] == bgd_class_name
    ok2 = table['slot'] == slot
    ok = ok1 * ok2
    if mag is not None:
        ok3 = table['mag'] == mag
        ok = ok * ok3

    times = table['time'][0]
    delta_t = times[1] - times[0]
        
    fig = plt.figure(figsize=(8, 25))

    data = []
    for i, aa in enumerate(table[ok]['deque_dict'][0][n_start:n_stop]):
        plt.subplot(12, 7, i + 1)
        row0 = table[ok]['row0'][0]
        col0 = table[ok]['col0'][0]
        index = n_start + i
        dat, im = plot_bgd_patch(aa, index, row0, col0, img_size, bgd_class_name,
                                vmin=vmin, vmax=vmax)
        plt.title('t{:4.0f}:\n{}, {}'.format(index * delta_t, row0[index], col0[index]));
        plt.axis('off')
        data.append(dat)

    plt.subplots_adjust(left=0.0, right=0.9,
                        top=0.9, bottom=0.2,
                        wspace=0.2, hspace=0.01)
    
    cbar_ax = fig.add_axes([0.92, 0.85, 0.01, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0, np.int(vmax/2), vmax])
    cbar.set_ticklabels(['0'.format(vmin), '{:0d}'.format(np.int(vmax/2)), '{}'.format(vmax)])


    print("Format of the titles is 'time: imgrow0, imgcol0'")
    print("Plot frames from {} to {}".format(n_start, n_stop))
    print("Bgd Class: {}, ndeque = {}".format(bgd_class_name, table[ok]['ndeque'][0]))

    return data
