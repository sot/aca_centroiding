import numpy as np
import collections


class FlightBgd(object):
    """
    Flight Background, current on-board algorithm:
    1 number representing average background for each frame
    """

    def __init__(self, bgdavg=0):
        self.img_size = 8
        self.bgdavg = bgdavg

    def __repr__(self):
        return ('<{} img_size={} bgdavg={}>'
                .format(self.__class__.__name__, self.img_size, self.bgdavg))

    def get_background(self):
        return np.ones((self.img_size, self.img_size)) * self.bgdavg # 8x8


class DynamBgd_Median(object):
    """
    Dynamic Background:
    median of n samples for edge pixels, bgdavg for non-edge pixels

    For each image, pick out the 'X's

    Currently implemented: options *8x8* and *6x6*,
    redefine r, c in get_centroids to compute **8x8**

    *8x8*                  *6x6* 32 telemetered vals for Science obs.
                                 (note that on-board we still have 8x8)
      (1)--->
    (3) XXXXXXXX (4)       .XXXX.
     |  X......X  |        X....X
     |/ X......X  |/       X....X
        X......X           X....X
        X......X           X....X
        X......X           .XXXX.
        X......X
        XXXXXXXX
        (2)--->

    OR

    **8x8**

       (1)--->
    (3) XXXXXXXX (4)
     |  XXXXXXXX  |
     |/ XX....XX  |/
        XX....XX
        XX....XX
        XX....XX
        XXXXXXXX
        XXXXXXXX
        (2)--->

    """

    def __init__(self, img_size, ndeque=1000, deque_dict=None, img=None,
                 row0=0, col0=0, bgdavg=0, max_bgd_excess=None, quench=-1):

        self.img_size = img_size
        self.ndeque = ndeque
        self.deque_dict = deque_dict
        if self.deque_dict is None:
            self.deque_dict = collections.OrderedDict()
        self.img = img
        self.row0 = row0
        self.col0 = col0
        self.bgdavg = bgdavg
        self.max_bgd_excess = max_bgd_excess
        if quench not in [-1, 0, 1]:
            raise ValueError("DynamBgd:: quench expected to be in [-1, 0, 1]")
        else:
            self.quench = quench

        if self.img_size == 8:
            self.r = [0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7,
                      1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
            self.c = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                      0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7]
        else:
            # on ground
            self.r = [0, 0, 0, 0, 5, 5, 5, 5, 1, 2, 3, 4, 1, 2, 3, 4]
            self.c = [1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 0, 0, 5, 5, 5, 5]

        self.edge_description = '1px outmost edge'


    def __repr__(self):
        return ('<{} img_size={} ndeque={} row0={} col0={} bgdavg={} '
                'edge_description=\'{}\'>'.format(self.__class__.__name__,
                                              self.img_size,
                                              self.ndeque,
                                              self.row0,
                                              self.col0,
                                              self.bgdavg,
                                              self.edge_description))


    def get_background(self):
        """
        Build a 8x8 background image:
        * for pixels that have been sampled use median/sigmaclip/etc
        * for pixels that have not been sampled use average background, bgdavg

        Note: if code calls get_background() more than 1 time, deque will be
              updated again! (correct it?)

        :returns: 8x8 array with bgd pixel values
        """

        if self.img is None:
            raise ValueError("get_background:: "
                             "Can't compute background, img is None")

        bgd_img = np.zeros((8, 8))

        self.deque_dict = self._update_deque_dict()

        for rr in range(self.img_size):
            for cc in range(self.img_size):
                key = (rr + self.row0, cc + self.col0)
                if key in self.deque_dict.keys():
                    bgd_img[rr, cc] = self._get_value(key)
                else:
                    bgd_img[rr, cc] = self.bgdavg

        if self.max_bgd_excess is not None:
            if self.quench==0:
                #
            elif self.quench==1:
                #
                    
        return bgd_img


    def _update_deque_dict(self):
        """
        Update dictionary which stores sampled edge pixel values in deques.
        Dictionary keys are tuples containing absolute pixel coordinates
        (in range -512:511, e.g. (120, 210)).


        Algorithm:
        1. Compute current coordinates of the edge pixels.
        2. Append or init deque with edge value if we are on the edge.
        3. Store up to ndeque values, if > ndeque elements in a deque, pop
           the first one (popleft).
        """

        r_current_edge = self.r + self.row0 * np.ones(len(self.r))
        c_current_edge = self.c + self.col0 * np.ones(len(self.c))

        r_current_edge = (r_current_edge).astype(int)
        c_current_edge = (c_current_edge).astype(int)

        edge_vals = self.edge_pixel_vals()
        deque_dict = self.deque_dict

        for (rr, cc) in zip(r_current_edge, c_current_edge):
            val = edge_vals[(rr, cc)]
            if (rr, cc) in deque_dict.keys():
                deque_dict[(rr, cc)].append(val)
                # Keep the length at ndeque
                if len(deque_dict[(rr, cc)]) > self.ndeque:
                    deque_dict[(rr, cc)].popleft()
            else: # initialize
                deque_dict[(rr, cc)] = collections.deque([val])
        return deque_dict


    # Pick out the edge pixels
    def edge_pixel_vals(self):
        """
        Pick up edge pixel values from the current image (self.img)

        :returns: dictionary with with keys being pixel coordinates,
                  e.g. (210, 124), and vals being pixel values
        :rtype: Ordered Dictionary
        """

        vals = collections.OrderedDict()

        self.img = self.img.reshape(8, 8)

        for rr, cc in zip(self.r, self.c): # (r, c) define location of edge pixels (X's)
            r_abs = self.row0 + rr
            c_abs = self.col0 + cc
            key = (r_abs, c_abs) # e.g. (210, 124), a tuple
            vals[key] = self.img[rr, cc]

        return vals


    def _get_value(self, key):
        """
        Return median
        """        
        return np.median(self.deque_dict[key])


class DynamBgd_SigmaClip(DynamBgd_Median):
    """
    Dynamic Background:
    Mean of (ndeque - 2) samples of a given pixel (after discarding min and
    max values), avgbgd for pixels that have not been sampled yet.
    """

    def _get_value(self, key):
        """
        Mask min and max, return mean of the unmasked values
        """
        deque = self.deque_dict[key]
        if len(deque) > 2:
            d_min = np.argmin(deque)
            d_max = np.argmax(deque)
            m = np.zeros(len(deque))
            m[d_min] = 1
            m[d_max] = 1
            deque = np.ma.array(deque, mask=m)
        return np.mean(deque)


class StaticHotPixel(object):
    """
    Static Hot Pixel Class

    :param val: hot pixel mean value
    :param sigma: standard deviation of hot pixel value assuming
                  Gaussian noise
    :param size: size of array containing hot pixel values,
                 equal to number of time frames

    Example:
    hot_pixel = StaticHotPixel(val=100, sigma=25, size=1000)

    hot_pixel.hp is an array of size=1000 with values normally distributed
    around val=100 with standard deviation sigma=25
    """

    def __init__(self, val=0, sigma=1, size=1):

        self.val = val
        self.sigma = sigma
        self.size = size
        self.hp = self.get_hot_pixel()

    def __repr__(self):
        return ('<{} val={} sigma={} size={}>'.format(self.__class__.__name__,
                                                      self.val,
                                                      self.sigma,
                                                      self.size))

    def get_hot_pixel(self):
        return np.random.normal(loc=self.val, scale=self.sigma, size=self.size)


class FlickeringHotPixel(object):
    """
    Flickering Hot Pixel Class

    :param vals: hot pixels flickers between two values defined by
                 vals=[val1, val2]
    :param sigmas: standard deviations of the hot pixels vals,
                   sigmas=[sigma1, sigma2]
    :param flicker_times: defines times at which hot pixel value switches
                          between vals, flicker_times=[t1, t2, ..., tn]
    :param size: size of array containing hot pixel values,
                 equal to the number of time frames

    Example:
    hot_pixel = FlickeringHotPixel(vals=[val1, val2], sigmas=[sigma1, sigma2],
                                   flicker_times=[250, 500, 750], size=1000)

    hot_pixel.hp is an array of size=1000 with values normally distributed
    around val1 (or val2) with standard deviation sigma1 (or sigma2).
    """

    def __init__(self, vals, sigmas, flicker_times, size=1):

        if not np.size(vals) == 2 or not np.size(vals) == np.size(sigmas):
            raise ValueError("FlickeringHotPixel:: vals and sigmas expected "
                             "to have the same size > 1")

        if not isinstance(flicker_times, collections.Iterable):
            raise TypeError("FlickeringHotPixel:: flicker_times expected "
                            " to be Iterable")

        if not np.size(flicker_times) > 0:
            raise ValueError("FlickeringHotPixel:: flicker_times expected "
                             "to have size > 0")

        self.vals = vals
        self.sigmas = sigmas
        self.flicker_times = flicker_times
        self.size = size
        self.hp = self.get_hot_pixel()


    def get_hot_pixel(self):
        """
        A variation of a square wave function with value changin between
        val1 and val2 at flicker_times.
        """

        hps = []
        for val, sigma in zip(self.vals, self.sigmas):
            hp = StaticHotPixel(val=val, sigma=sigma, size=self.size)
            hps.append(hp)

        ftimes = self.flicker_times
        if not ftimes[0] == 0:
            ftimes = [0] + ftimes
        if not ftimes[-1] == self.size:
            ftimes.append(self.size)

        filt = []
        a = True
        for i, ftime in enumerate(ftimes[1:]):
            filt = filt + [a] * (ftime - ftimes[i])
            a = not a

        hp = np.array(filt) * np.array(hps[0].hp) + \
             np.logical_not(filt) * np.array(hps[1].hp)

        return hp




