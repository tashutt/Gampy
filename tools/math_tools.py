def rotate_ray(input_ray, rotation_target):
    """
    Rotates input_ray by angle between [0,0,1] and rotation_target, to create
    output_ray, which is returned.

    First index is space 0:2, second is ray number.

    Based on Rodrique's rotation formula
    (https://en.wikipedia.org/wiki/Rodrigues#27_rotation_formula)

    11/3/2021 TS - port of 7/15/18 matlab routine
    """

    import numpy as np

    #   These used in calculation.  Following standard notation,
    #   a rotates into b.  Here a is (0,0,1) and b is rotation_target
    #   k is axis of rotation
    if input_ray.ndim > 1:
        a = np.tile(np.array([[0], [0], [1]]), len(input_ray[0, :]))
    else:
        input_ray = input_ray.reshape(3,1)
        a = np.array([[0], [0], [1]])
    b = rotation_target
    if b.size==3:
        b = b.reshape(3, 1)
        if input_ray.size>3:
            b = np.tile(b, len(input_ray[0, :]))

    #   Can't rotate these: have normal aligned with [0 0 1], hence
    #   cross product is zero.
    straight_up = np.abs(b[2, :] - 1) < 1e-7
    straight_down = np.abs(b[2, :] + 1) < 1e-7
    can_rotate = ~(straight_up | straight_down)

    a_cross_b = np.cross(a.transpose(), b.transpose()).transpose()
    a_cross_b_dot = va_dot(a_cross_b, a_cross_b)

    cos_theta = va_dot(a, b)
    sin_theta = np.sqrt(a_cross_b_dot)

    k = np.zeros_like(a_cross_b)
    k[:, can_rotate] = a_cross_b[:, can_rotate] / sin_theta[can_rotate]

    #   Pre-assign to deal with exception.  straight_up now done
    output_ray = input_ray

    #   Rotate those that can
    if np.sum(can_rotate) > 0:
        output_ray[:, can_rotate,] \
            = input_ray[:, can_rotate,] \
                * np.tile(cos_theta[can_rotate], (3, 1)) \
                + np.cross(
                    k[:, can_rotate].transpose(),
                    input_ray[:, can_rotate].transpose()
                    ).transpose() \
                * np.tile(sin_theta[can_rotate], (3, 1)) \
                + k[:, can_rotate] \
                * np.tile(
                    va_dot(k[:, can_rotate], input_ray[:, can_rotate]),
                    (3, 1)) \
                * (1 - np.tile(cos_theta[can_rotate], (3, 1)))

    #   straight_down gets z component flipped
    if np.sum(straight_down) > 0:
        output_ray[2, straight_down] = -output_ray[2, straight_down]

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
