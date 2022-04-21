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

