def rotate_ray(input_ray, rotation_target):
    """
    Rotates input_ray by angle between [0,0,1] and rotation_target, to create
    output_ray, which is returned.

    Either input can be a vector or array, but arrays must be same size.

    Note: non-normalized vectors (or spatial direction of arryas) give
        incorrect results

    Based on Rodrique's rotation formula
    (https://en.wikipedia.org/wiki/Rodrigues#27_rotation_formula)

    11/3/2021 TS - port of 7/15/18 matlab routine
    4/24 TS - allow flexibility on input shapes
    """

    import numpy as np
    import copy
    import sys

    #   Check inputs, and adjust as needed.

    #   Both must have a size divisible by 3
    if (input_ray.size % 3 != 0) | (rotation_target.size % 3 != 0):
        sys.exit('in math_tools.rotate_ray: bad inputs')

    #   Each must be either a 3 vectors or an array
    if (input_ray.size>3) & (input_ray.ndim!=2):
        sys.exit('in math_tools.rotate_ray: bad input_ray')
    if (rotation_target.size>3) & (rotation_target.ndim!=2):
        sys.exit('in math_tools.rotate_ray: bad rotation_target')

    #   Arrays cannot be of unequal size
    if ((rotation_target.ndim!=2) & (input_ray.ndim!=2))\
        & (input_ray.size!=rotation_target.size):
        sys.exit('in math_tools.rotate_ray: bad inputs')

    #   Make float if not
    if input_ray.dtype=='int':
        input_ray = np.array(input_ray, dtype=float)
    if rotation_target.dtype=='int':
        rotation_target = np.array(rotation_target, dtype=float)

    #   For arrays, enforce that space axis is second
    transposed_input = False
    if input_ray.size>3:
        if input_ray.shape[1]!=3:
            input_ray = input_ray.T
            transposed_input = True
    if rotation_target.size>3:
        if rotation_target.shape[1]!=3:
            rotation_target = rotation_target.T

    #   If both are vectors, some ugliness: make both array of
    #   size (2, 3).
    padded_input = False
    if (input_ray.size==3) & (rotation_target.size==3):
        input_ray = np.broadcast_to(input_ray.reshape(3), (2, 3))
        rotation_target = np.broadcast_to(rotation_target.reshape(3), (2, 3))
        padded_input = True

    #   If one is vector and other array, broadcast vector to array
    if (input_ray.size==3) & (rotation_target.size>3):
        input_ray = np.broadcast_to(
            input_ray.reshape(3),
            (rotation_target.shape[0], 3)
            )
    elif (input_ray.size>3) & (rotation_target.size==3):
        rotation_target = np.broadcast_to(
            rotation_target.reshape(3),
            (input_ray.shape[0], 3)
            )

    #   rotation_target must be normalized
    if np.any(np.abs((rotation_target * rotation_target).sum(axis=1)-1.0)
              >1e-15):
        sys.exit('in math_tools.rotate_ray: rotation_target not normalized')

    #   Following standard notation, the rotation is defined as
    #   rotating a to b. Here a is (0,0,1) and b is rotation_target.
    #   k is axis of rotation
    if input_ray.size > 3:
        a = np.broadcast_to(np.array((0., 0., 1.)), (input_ray.shape[0], 3))
    else:
        a = np.array((0, 0, 1))
    b = rotation_target

    a_cross_b = np.cross(a, b)
    a_cross_b_dot = (a_cross_b * a_cross_b).sum(axis=1)

    cos_theta = (a * b).sum(axis=1)
    #   TODO: why sqrt here, and not on cos_theta?
    sin_theta = np.sqrt(a_cross_b_dot)

    #   Can't rotate these: have normal aligned with [0 0 1], hence
    #   cross product is zero.
    straight_up = np.abs(b[:, 2] - 1) < 1e-7
    straight_down = np.abs(b[:, 2] + 1) < 1e-7
    can_rotate = ~(straight_up | straight_down)

    k = np.zeros_like(a_cross_b)
    k[can_rotate, :] = a_cross_b[can_rotate, :] \
        / sin_theta[can_rotate][:, None]

    #   Pre-assign to deal with exception.  straight_up now done
    output_ray = copy.copy(input_ray)

    #   Rotate those that can
    ncr = can_rotate.sum()
    if np.sum(can_rotate) > 0:
        output_ray[can_rotate, :] = (
            input_ray[can_rotate, :]
            * np.broadcast_to(
                cos_theta[can_rotate, None],
                (ncr, 3)
                )
            + np.cross(k[can_rotate, :], input_ray[can_rotate, :])
            * np.broadcast_to(
                sin_theta[can_rotate][:, None],
                (ncr, 3)
                )
            + k[can_rotate, :]
            * np.broadcast_to(
                (k[can_rotate, :] * input_ray[can_rotate, :]).sum(axis=1)
                    [:, None],
                (ncr, 3)
                )
            * np.broadcast_to(
                (1 - cos_theta[can_rotate])[:, None],
                (ncr, 3)
                )
            )

    #   straight_down gets z component flipped
    if np.sum(straight_down) > 0:
        output_ray[straight_down, 2] = -output_ray[straight_down, 2]

    #   If input needed reformatting, undo.
    if transposed_input:
        output_ray = output_ray.T
    if padded_input:
        output_ray = output_ray[0, :]

    return output_ray

def sph2cart(theta, phi, r=1.0):
    """ returns cartersion vector (or vector array) s, from
        spherical coordinate inputs theta, phi.   Theta is angle from
        z-axis.  Note - similar, but not identical to matlab equivalent.
    """
    import numpy as np

    rho = np.sin(theta)
    if len(theta)>1:
        s = np.zeros((3, len(theta)))
        s[0, :] = r * rho * np.cos(phi)
        s[1, :] = r * rho * np.sin(phi)
        s[2, :] = r * np.cos(theta)
    else:
        s = np.zeros(3)
        s[0] = r * rho * np.cos(phi)
        s[1] = r * rho * np.sin(phi)
        s[2] = r * np.cos(theta)

    return s

#   dot product
def dot(a, b):
    """ Ordinary vector dot product"""
    import numpy as np
    return  np.sum(a * b, axis=0)

def va_dot(array_1, array_2):
    """ Computes ordinary 3D vector dot product for each element of
        either vectors or arrays with dimensions [3, array_length]
    """
    if array_1.size>3:
        dot_product = array_1[0, :] * array_2[0, :]
        dot_product += array_1[1, :] * array_2[1, :]
        dot_product += array_1[2, :] * array_2[2, :]
    else:
        dot_product = sum(array_1 * array_2)

    return dot_product

def generate_circles(theta, direction_ray, num_points):
    """ Generates cirles on unit sphere in cartesian coordinates

        theta - vector of cone opening angles
        direction_ray - array of rays to center of circles
        num_points - number of uniformly spaced points on circle

        2/23 TS - port from matlab
    """

    import numpy as np
    from math import pi

    #   Create circle of points about (0,0,1) representing cones
    phi = np.linspace(0, 1, num_points) * 2 * pi
    cone_radius = np.sin(theta)
    x = np.outer(np.cos(phi), cone_radius)
    y = np.outer(np.sin(phi), cone_radius)

    z = np.ones_like(x) * np.cos(theta)

    #   Now rotate to final location.  Awkwardly, need to do this per point of
    #   circle
    circles = np.zeros((3, theta.size, num_points))
    s = np.zeros((3, theta.size))
    for n in range(num_points):
        s[0, :] = x[n, :]
        s[1, :] = y[n, :]
        s[2, :] = z[n, :]
        #   Now rotate to proper position
        circles[:, :, n] = rotate_ray(
            s,
            direction_ray
                / np.sqrt(dot(direction_ray, direction_ray))
            )

    return circles
