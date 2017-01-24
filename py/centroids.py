import numpy as np
from mica.archive import aca_l0
from chandra_aca import transform
from copy import deepcopy
import numpy.ma as ma
from astropy.table import Table
# local imports:
import classes
from classes import *


def centroids(slot, slot_data, img_size, bgd_object, nframes=1000):
    """
    Returns a list of dictionaries.

    Each dictionary contains the following keys:

    * 'slot': slot number
    * 'time': time axis, array of size = nframes
    * 'imgrow0': IMGROW0, array of size = nframes
    * 'imgcol0': IMGCOL0, array of size = nframes
    * 'imgraw': raw images, array of n=nframes 8x8 images
    * 'bgdimg': bgd images, array of n=nframes 8x8 bgd images
    * 'deque_dicts': n=nframes dictionaries with keys being pixel coordinates
                     and vals being arrays of up to ndeque pixel value samples
    * 'yan': derived yan centroids, array of size = nframes
    * 'zan': derived zan centroids, array of size = nframes
    * 'row': derived row centroids (from 0 to 8), array of size = nframes
    * 'col': derived column centroids (from 0 to 8), array of size = nframes

    Calls:
        get_centroids

    """

    rows = []

    #print('Slot = {}'.format(slot))

    slot_row = {}

    # rowcol_cntrds, yagzag_cntrds, bgd_imgs, deque_dicts
    result =  get_centroids(slot_data, img_size, bgd_object, nframes=nframes)

    slot_data = Table(slot_data)

    slot_row['slot'] = slot
    slot_row['time'] = slot_data['TIME'][:nframes]
    slot_row['row0'] = slot_data['IMGROW0'][:nframes]
    slot_row['col0'] = slot_data['IMGCOL0'][:nframes]
    # imgraw is not bgd subtracted!
    slot_row['imgraw'] = slot_data['IMGRAW'][:nframes] # 64
    slot_row['bgdimg'] = result[2] # 8x8
    slot_row['img'] = slot_data['IMGRAW'][:nframes].reshape(nframes, 8, 8) - result[2] # 8x8
    slot_row['deque_dict'] = result[3]

    yagzag = np.array(result[1]).T[0] # [0] - yag, [1] - zag

    slot_row['yan'] = yagzag[0]
    slot_row['zan'] = yagzag[1]

    rowcol = np.array(result[0]).T # [0] - row, [1] - col, 1:7

    slot_row['row'] = rowcol[0]
    slot_row['col'] = rowcol[1]

    rows.append(slot_row)

    return rows


def get_centroids(slot_data, img_size, bgd_object, nframes=None):
    """
    For each frame:
    1. Compute background image using get_background() method of bgd_object.
       Algorithm depends on the class of bgd_object
            - FlightBgd: current on-board algorithm
            - DarkCurrent_Median_Bgd: median for sampled pixels, average bgd
              for not sampled pixels
            - DarkCurrent_SigmaClip_Bgd: sigma clipping for sampled pixels,
              avg bgd for not sampled pixels
    2. Subtract background
    3. Compute centroids in image coordinates 0:8 (variable name: row/col)
    4. Transform to get yagzag coordinates

    Returns:
    * rowcol_centroids: list of nframes centroid pairs,
                        [[row centroid1, col centroid1], [], ...]
    * yanzan_centroids: list of nframes centroid pairs,
                        [[yag centroid1, zag centroid1], [], ...]
    * bgd_imgs: list of nframes 8x8 arrays representing background images
    * deque_dicts: list of nframes dictionaries with keys being pixel
                   coordinates, e.g. (210, 130), and vals being deques of
                   ndeque pixel value samples

    :param slot_data: simulated or flight aca_l0 slot data
    :param img_size: image size, either 6 or 8 (pixels)
    :param bgd_object: background object defined in classes.py
    :param nframes: number of time frames
    """

    if img_size not in [6, 8]:
        raise ValueError('get_centroids:: expected img_size = 6 or 8')

    if nframes is None:
        nframes = len(slot_data)

    yagzag_centroids = []
    rowcol_centroids = []
    bgd_imgs = []
    deque_dicts = []

    for index in range(0, nframes):

        frame_data = slot_data[index:index + 1]

        bgd_object.bgdavg = frame_data['BGDAVG'][0]

        if isinstance(bgd_object, (DynamBgd_Median, DynamBgd_SigmaClip)):
            bgd_object.img = frame_data['IMGRAW'][0]
            bgd_object.row0 = frame_data['IMGROW0'][0]
            bgd_object.col0 = frame_data['IMGCOL0'][0]

        bgd_img = bgd_object.get_background() # 8x8

        bgd_imgs.append(bgd_img)

        if isinstance(bgd_object, (DynamBgd_Median, DynamBgd_SigmaClip)):
            deque_dict = bgd_object.deque_dict
            deque_dicts.append(deepcopy(deque_dict))

        raw_img = frame_data['IMGRAW'][0].reshape(8, 8)

        img = raw_img - bgd_img

        # If dynamic background, don't oversubtract?
        # Value of pixel with bgd > raw image value will be set to zero:
        
        #if isinstance(bgd_object, (DynamBgd_Median, DynamBgd_SigmaClip)):
        #    bgd_mask = bgd_img > raw_img
        #    img = raw_img - ma.array(bgd_img, mask=bgd_mask)
        #    img = img.data * ~bgd_mask

        # Calculate centroids for current bgd-subtracted img, use first moments
        rowcol = get_current_centroids(img, img_size)
        rowcol_centroids.append(rowcol)

        # Translate (row, column) centroid to (yag, zag)
        y_pixel = rowcol[0] + frame_data['IMGROW0']
        z_pixel = rowcol[1] + frame_data['IMGCOL0']
        yagzag = transform.pixels_to_yagzag(y_pixel, z_pixel)

        yagzag_centroids.append(yagzag)

    return rowcol_centroids, yagzag_centroids, bgd_imgs, deque_dicts


def get_current_centroids(img, img_size):
    """
    Compute centroids using first moments.

    :param img: bgd-subtracted image, always 8x8
    :param img_size: image size
    """

    if img_size == 8:
        img_mask = get_mask_8x8_centered()
    else:
        img_mask = None

    if img_mask is not None:
        img = ma.array(img, mask=img_mask)

    num = np.arange(0.5, 6.5)

    if (img_size == 8):
        # ER observations
        img = zero_6x6_corners(img, centered=True)
    else:
        # Science observations
        img = zero_6x6_corners(img, centered=False)

    centroids = []
    for ax in [1, 0]: # [row, col]
        # Def of flat is where img_mask becomes relevant for ER data
        flat = np.sum(img, axis=ax)
        if (img_size == 6):
            centroid = np.sum(flat[:-2] * num) / np.sum(flat[:-2]) # 0:6
        else:
            # 1:7, is +1 relevant?
            # yes, if row0/col0 always the lower left pixel in 8x8
            centroid = np.sum(flat[1:-1] * num) / np.sum(flat[1:-1]) + 1 # 1:7
        centroids.append(centroid)

    return centroids


def zero_6x6_corners(img, centered=True):
    """
    Set 6x6 corners to zero.

    :param img: 8x8 array
    :param centered: True for 6x6 image centered in 8x8 array (ER obs),
                     False for 6x6 image occupying the [0:5,0:5] part of
                     8x8 array (Science obs)
    """
    if not img.shape == (8, 8):
        raise ValueError("zero_6x6_corners:: Img should be an 8x8 array")
    if centered:
        r4 = [1, 1, 6, 6]
        c4 = [1, 6, 1, 6]
    else:
        r4 = [0, 0, 5, 5]
        c4 = [0, 5, 0, 5]
    for rr, cc in zip(r4, c4):
        img[rr][cc] = 0.0
    return img


def get_mask_8x8_centered():
    """
    In 8x8 img (ER obs), mask the edge pixels, leave r/c 1:7 unmasked.
    For science observations, raw images are masked by default
    (r/c 0:6 are left unmasked).
    """

    m = """\
        1 1 1 1 1 1 1 1
        1 0 0 0 0 0 0 1
        1 0 0 0 0 0 0 1
        1 0 0 0 0 0 0 1
        1 0 0 0 0 0 0 1
        1 0 0 0 0 0 0 1
        1 0 0 0 0 0 0 1
        1 1 1 1 1 1 1 1"""
    mask = np.array([line.split() for line in m.splitlines()], dtype=float)
    return mask

