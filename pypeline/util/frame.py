r"""
Definition of the UVW frame.
"""

import numpy as np
import astropy.coordinates as aspy


def uvw_basis(field_center: aspy.SkyCoord) -> np.ndarray:
    r"""
    Transformation matrix associated to the local UVW frame.

    Parameters
    ----------
    field_center: astropy.coordinates.SkyCoord
        Center of the FoV to which the local frame is attached.

    Returns
    -------
    uvw_frame: np.ndarray
        (3, 3) transformation matrix. Each column contains the ICRS coordinates of the U, V and W basis vectors defining the frame.
    """
    field_center_lon, field_center_lat = field_center.data.lon.rad, field_center.data.lat.rad
    field_center_xyz = field_center.cartesian.xyz.value
    # UVW reference frame
    w_dir = field_center_xyz
    u_dir = np.array([-np.sin(field_center_lon), np.cos(field_center_lon), 0])
    v_dir = np.array(
        [-np.cos(field_center_lon) * np.sin(field_center_lat), -np.sin(field_center_lon) * np.sin(field_center_lat),
         np.cos(field_center_lat)])
    uvw_frame = np.stack((u_dir, v_dir, w_dir), axis=-1)
    return uvw_frame

def xyz_at_latitude(local_xyz, lat):
    """
    Rotate local XYZ coordinates into celestial XYZ coordinates. These
    coordinate systems are very similar, with X pointing towards the
    geographical east in both cases. However, before the rotation Z
    points towards the zenith, whereas afterwards it will point towards
    celestial north (parallel to the earth axis).
    :param lat: target latitude (radians or astropy quantity)
    :param local_xyz: Array of local XYZ coordinates
    :return: Celestial XYZ coordinates
    """

    # return enu_to_eci(local_xyz, lat)
    x, y, z = np.hsplit(local_xyz, 3)  # pylint: disable=unbalanced-tuple-unpacking

    lat2 = np.pi / 2 - lat
    y2 = -z * np.sin(lat2) + y * np.cos(lat2)
    z2 =  z * np.cos(lat2) + y * np.sin(lat2)

    return np.hstack([x, y2, z2])

def xyz_to_uvw(xyz, ha, dec):
    """
    Rotate :math:`(x,y,z)` positions in earth coordinates to
    :math:`(u,v,w)` coordinates relative to astronomical source
    position :math:`(ha, dec)`. Can be used for both antenna positions
    as well as for baselines.
    Hour angle and declination can be given as single values or arrays
    of the same length. Angles can be given as radians or astropy
    quantities with a valid conversion.
    :param xyz: :math:`(x,y,z)` co-ordinates of antennas in array
    :param ha: hour angle of phase tracking centre (:math:`ha = ra - lst`)
    :param dec: declination of phase tracking centre.
    """

    # return eci_to_uvw(xyz, ha, dec)
    x, y, z = np.hsplit(xyz, 3)  # pylint: disable=unbalanced-tuple-unpacking

    # Two rotations:
    #  1. by 'ha' along the z axis
    #  2. by '90-dec' along the u axis
    u = x * np.cos(ha) - y * np.sin(ha)
    v0 = x * np.sin(ha) + y * np.cos(ha)
    w = z * np.sin(dec) - v0 * np.cos(dec)
    v = z * np.cos(dec) + v0 * np.sin(dec)

    return np.hstack([u, v, w])