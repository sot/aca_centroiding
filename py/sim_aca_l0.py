import numpy as np
from chandra_aca import transform
import collections
import numpy.ma as ma
# local imports:
import centroids as cntr


def simulate_aca_l0(img_size, hot_pixels=None, nframes=1000,
                    delta_t=4.1, integ=1.696, bgdavg=None,
                    fwhm=1.8, mag=10.2,
                    delta=-0.25,  ampl=8., period=1000., phase=0.):
    """
    Simulate star image in aca_l0 format:

    * simulate yaw and pitch dither pattern, integerize it, and derive
      simulated imgrow0 and imgcol0.
    * compute row and column offsets of star from image center
    * simulate star with magnitude mag, fwhm, located at


    :param img_size: size of the image = 6 or 8 (pixels)
    :param nframes: number of time frames
    :param hot_pixels: dictionary with keys being pixel coordinates,
                       e.g. (120, 210) (row, column), and vals being arrays
                       of size=nframes containing hot pixel values.
    :param delta_t: readout time for each image (in sec)
    :param integ: integration time for each image (in sec)
    :param bgdavg: average value of flight background, here it is a number,
                   in general this should be an array of size nframes
                   (then use bgd=bgdavg[i] in line 37)
    :param fwhm: FWHM of a typical ACA star (in pixels)
    :param mag: star magnitude
    :param ampl: dither amplitude (in arcsec), a number or a 2-element list
                 [yaw_ampl, pitch_ampl]
    :param period: dither period (in sec), a number or a 2-element list
                   [yaw_period, pitch_period]
    :param phase: dither phase (a fraction of 2pi), a number or a 2-element
                  list [yaw_phase, pitch_phase]
    :param delta:
    """
   
    if hot_pixels is not None:
        sizes = [np.size(val) for val in hot_pixels.values()]
        if any(s != nframes for s in sizes):
            raise TypeError("simulate_aca_l0:: "
                            "hot_pixel.hp size expected to be {}".format(nframes))

    times = np.arange(nframes) * delta_t

    row_drift, col_drift = get_rowcol_drift(times,
                                            ampl=ampl,
                                            period=period,
                                            phase=phase,
                                            delta=delta)

    imgrow0 = np.array(np.round(row_drift + delta), dtype=np.int)
    imgcol0 = np.array(np.round(col_drift + delta), dtype=np.int)


    # star row offset (col offset) from image center, defined as difference
    # between row_drift (col_drift) and its integerized value imgrow0 (imgcol0)
    star_roff = row_drift - imgrow0
    star_coff = col_drift - imgcol0

    data = []
    img_size2 = img_size * img_size

    for i, time in enumerate(times):
        imgraw = simulate_star(fwhm, mag, integ, bgd=bgdavg,
                               roff=star_roff[i], coff=star_coff[i]) # 8x8

        # add hot pixels if defined and if they fit in the current 8x8 image
        if hot_pixels is not None:
            for key, val in hot_pixels.items():
                rr = key[0] - imgrow0[i]
                cc = key[1] - imgcol0[i]
                if rr in range(8) and cc in range(8):
                    imgraw[rr, cc] = imgraw[rr, cc] + val[i]

        imgraw = imgraw.reshape(1, img_size2)[0]
        mask = img_size2 * [0]
        fill_value = 1.e20
        imgraw = ma.array(data=imgraw, mask=mask, fill_value=fill_value)

        data_row = (time, imgraw, imgrow0[i], imgcol0[i], bgdavg, img_size)
        data.append(data_row)

    data = np.ma.array(data, dtype=[('TIME', '>f8'), ('IMGRAW', '>f4', (64,)),
                                    ('IMGROW0', '>i2'), ('IMGCOL0', '>i2'),
                                    ('BGDAVG', '>i2'), ('IMGSIZE', '>i4')])

    # True centroids, center of 8x8 image + offsets, check for 6x6?
    halfsize = np.int(img_size / 2)
    true_centroids = np.array([halfsize + star_roff, halfsize + star_coff])

    return data, true_centroids


def get_rowcol_drift(times, ampl=8., period=1000., phase=0., delta=-0.25):
    """
    Compute impact of dither pattern on rows and columns fitting
    in an image window. Needed to find imgrow0 and imgcol0 that change
    response to dither to keep star in center of image window.

    :param times: time axis
    :param ampl: dither amplitude (arcsec), a number or a 2-element list
                 [ampl_yaw, ampl_pitch]
    :param period: dither period (sec), a number or a 2-element list
                   [period_yaw, period_pitch]
    :param phase: dither phase (fraction of 2pi), a number or a 2-element
                  list [phase_yaw, phase_pitch]
    :param delta: see validations.ipynb
    """

    ampl_yaw, ampl_pitch = do_assignment(ampl)
    period_yaw, period_pitch = do_assignment(period)
    phase_yaw, phase_pitch = do_assignment(phase)

    yaw = ampl_yaw * np.sin(2 * np.pi * times / period_yaw +
                            2 * np.pi * phase_yaw)
    pitch = ampl_pitch * np.sin(2 * np.pi * times / period_pitch +
                                2 * np.pi * phase_pitch)

    pxsize = 5. # arcsec per pixel

    row_drift = yaw / pxsize # in pixels
    col_drift = pitch / pxsize

    return (row_drift, col_drift)


def do_assignment(val):
    """
    Check if val is a number or 2-d array.
    """
    if isinstance(val, collections.Iterable):
        if len(val) == 2 and all([isinstance(i, (int, float)) for i in val]):
            val1, val2 = val
        else:
            raise TypeError("do_assignment:: "
                            "Expecting an iterable with 2 numeric elements")
    elif isinstance(val, (int, float)):
        val1 = val2 = val
    else:
        raise TypeError("do_assignemnt:: "
                        "Expecting value to be a number or an iterable")
    return (val1, val2)


def simulate_star(fwhm, mag, integ, bgd=None, roff=0, coff=0):
    """
    Simulate a 2-d Gaussian star with magnitude mag, fwhm and gaussian noise,
    located in the center of a 8x8 image with row and column offsets roff and
    coff (in pixels).

    :param fwhm: star's full width at half maximum in pixels
    :param mag: magnitude
    :param integ: image integration time
    :param bgd: background value (int, as in current on-board algorithm)
    :param roff: row offset from the center of a 8x8 px image (in pixels)
    :param coff: column offset from the center of a 8x8 px image (in pixels)
    """

    img_size = 8
    img_size2 = img_size * img_size

    if not isinstance(bgd, (int, float)):
        raise TypeError("simulate_star:: bgd expected to be (int, float)")

    star = np.zeros((img_size, img_size))

    # Mag to counts conversion
    gain = 5. # e-/ADU
    counts = integ * transform.mag_to_count_rate(mag) / gain

    # Gaussian model
    halfsize = np.int(img_size / 2)
    row, col = np.mgrid[-halfsize:halfsize, -halfsize:halfsize] + 0.5
    sigma = fwhm / (2. * np.sqrt(2. * np.log(2.)))
    g = np.exp(-((row - roff)**2  / sigma**2 + (col - coff)**2 / sigma**2) / 2.)

    # Zero 6x6 corners
    g = cntr.zero_6x6_corners(g, centered=True)

    # Normalize to counts
    i1 = np.int(halfsize + 0.5 - 3)
    i2 = np.int(halfsize + 0.5 + 3)
    g = counts * g / g[i1:i2][i1:i2].sum()

    # Simulate star
    star = np.random.normal(g)

    # Add background
    if np.shape(bgd) == ():
        bgd = np.ones((img_size, img_size)) * bgd

    star = star + bgd

    return np.rint(star)


def dither_acis():
    """
    ACIS default dither parameters
    """
    ampl = 8. # arcsec
    yaw_period = 1000.0 # sec
    pitch_period = 707.1 # sec
    period = [yaw_period, pitch_period]
    return (ampl, period)


def dither_hrc():
    """
    HRC default dither parameters
    """
    ampl = 20. # arcsec
    yaw_period = 1087.0 # sec
    pitch_period = 768.6 # sec
    period = [yaw_period, pitch_period]
    return (ampl, period)
