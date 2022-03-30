#  from bluebild.bluebild import bluebild

__all__ = ["bluebild"]
__version__ = "0.1"

import numpy as np
from enum import IntEnum
import ctypes
from pathlib import Path
from ctypes import c_void_p, c_int, c_float, c_double
import platform

if platform.system() == 'Darwin':
    _bluebild_lib_name = 'libbluebild.dylib'
else:
    _bluebild_lib_name = 'libbluebild.so'

_bluebild_lib_paths = ['', 'lib', 'lib64', '../', '../lib', '../lib64', '../..', '../../lib', '../../lib64', '../../..', '../../../lib', '../../../lib64']

_bluebild_lib = None
_bluebild_module_loc = str(Path(__file__).absolute().parent.as_posix())

# try to find library installed with module
for p in _bluebild_lib_paths:
    if _bluebild_lib:
        break
    try:
        _bluebild_lib = ctypes.cdll.LoadLibrary(_bluebild_module_loc + '/' + p + '/' + _bluebild_lib_name)
    except Exception:
        pass

# try to find library in default system shared library search paths as fallback
if not _bluebild_lib:
    try:
        _bluebild_lib = ctypes.cdll.LoadLibrary(_bluebild_lib_name)
    except Exception:
        raise RuntimeError("Failed to find bluebild library")


_bluebild_ctx_create = _bluebild_lib.bluebild_ctx_create
_bluebild_ctx_create.argtypes = [c_int, ctypes.POINTER(c_void_p)]
_bluebild_ctx_create.restypes = c_int

_bluebild_ctx_destroy = _bluebild_lib.bluebild_ctx_destroy
_bluebild_ctx_destroy.argtypes = [ctypes.POINTER(c_void_p)]
_bluebild_ctx_destroy.restypes = c_int

_bluebild_eigh_s = _bluebild_lib.bluebild_eigh_s
_bluebild_eigh_s.argtypes = [
    c_void_p,
    c_int,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_void_p,
    c_void_p,
    c_int,
]
_bluebild_eigh_s.restypes = c_int

_bluebild_eigh_d = _bluebild_lib.bluebild_eigh_d
_bluebild_eigh_d.argtypes = [
    c_void_p,
    c_int,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_void_p,
    c_void_p,
    c_int,
]
_bluebild_eigh_d.restypes = c_int

_bluebild_sensitivity_field_data_s = _bluebild_lib.bluebild_sensitivity_field_data_s
_bluebild_sensitivity_field_data_s.argtypes = [
    c_void_p,
    c_float,
    c_int,
    c_int,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_void_p,
    c_int,
]
_bluebild_sensitivity_field_data_s.restypes = c_int

_bluebild_sensitivity_field_data_d = _bluebild_lib.bluebild_sensitivity_field_data_d
_bluebild_sensitivity_field_data_d.argtypes = [
    c_void_p,
    c_double,
    c_int,
    c_int,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_void_p,
    c_int,
]
_bluebild_sensitivity_field_data_d.restypes = c_int

_bluebild_intensity_field_data_s = _bluebild_lib.bluebild_intensity_field_data_s
_bluebild_intensity_field_data_s.argtypes = [
    c_void_p,
    c_float,
    c_int,
    c_int,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_void_p,
    c_int,
    c_int,
    c_void_p,
    c_void_p,
]
_bluebild_intensity_field_data_s.restypes = c_int

_bluebild_intensity_field_data_d = _bluebild_lib.bluebild_intensity_field_data_d
_bluebild_intensity_field_data_d.argtypes = [
    c_void_p,
    c_double,
    c_int,
    c_int,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_void_p,
    c_int,
    c_int,
    c_void_p,
    c_void_p,
]
_bluebild_intensity_field_data_d.restypes = c_int

_bluebild_gram_matrix_s = _bluebild_lib.bluebild_gram_matrix_s
_bluebild_gram_matrix_s.argtypes = [
    c_void_p,
    c_int,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_int,
    c_float,
    c_void_p,
    c_int,
]
_bluebild_gram_matrix_s.restypes = c_int

_bluebild_gram_matrix_d = _bluebild_lib.bluebild_gram_matrix_d
_bluebild_gram_matrix_d.argtypes = [
    c_void_p,
    c_int,
    c_int,
    c_void_p,
    c_int,
    c_void_p,
    c_int,
    c_double,
    c_void_p,
    c_int,
]
_bluebild_gram_matrix_d.restypes = c_int


_bluebild_nufft3d3_create_s = _bluebild_lib.bluebild_nufft3d3_create_s
_bluebild_nufft3d3_create_s.argtypes = [c_void_p, c_int, c_float, c_int, c_int, c_void_p, c_void_p, c_void_p, c_int, c_void_p, c_void_p, c_void_p, ctypes.POINTER(c_void_p)]
_bluebild_nufft3d3_create_s.restypes = c_int

_bluebild_nufft3d3_destroy_s = _bluebild_lib.bluebild_nufft3d3_destroy_s
_bluebild_nufft3d3_destroy_s.argtypes = [ctypes.POINTER(c_void_p)]
_bluebild_nufft3d3_destroy_s.restypes = c_int

_bluebild_nufft3d3_execute_s = _bluebild_lib.bluebild_nufft3d3_execute_s
_bluebild_nufft3d3_execute_s.argtypes = [c_void_p, c_void_p, c_void_p]
_bluebild_nufft3d3_execute_s.restypes = c_int

_bluebild_nufft3d3_create_d = _bluebild_lib.bluebild_nufft3d3_create_d
_bluebild_nufft3d3_create_d.argtypes = [c_void_p, c_int, c_double, c_int, c_int, c_void_p, c_void_p, c_void_p, c_int, c_void_p, c_void_p, c_void_p, ctypes.POINTER(c_void_p)]
_bluebild_nufft3d3_create_d.restypes = c_int

_bluebild_nufft3d3_destroy_d = _bluebild_lib.bluebild_nufft3d3_destroy_d
_bluebild_nufft3d3_destroy_d.argtypes = [ctypes.POINTER(c_void_p)]
_bluebild_nufft3d3_destroy_d.restypes = c_int

_bluebild_nufft3d3_execute_d = _bluebild_lib.bluebild_nufft3d3_execute_d
_bluebild_nufft3d3_execute_d.argtypes = [c_void_p, c_void_p, c_void_p]
_bluebild_nufft3d3_execute_d.restypes = c_int

class ProcessingUnit(IntEnum):
    AUTO = 0
    CPU = 1
    GPU = 2


class Context:
    def __init__(self, pu=ProcessingUnit.AUTO):
        self.ctx = ctypes.c_void_p(None)
        ier = _bluebild_ctx_create(int(pu), ctypes.byref(self.ctx))
        if ier != 0:
            raise RuntimeError("Bluebild: Failed to create context.")

    def __del__(self):
        if self.ctx:
            ier = _bluebild_ctx_destroy(ctypes.byref(self.ctx))
            if ier != 0:
                raise RuntimeError("Bluebild: Failed to destroy context.")

    def eigh(self, A, B=None, n_eig=0):
        if len(A.shape) != 2:
            raise RuntimeError("Bluebild: Input must be 2 dimensional.")

        if A.shape[0] != A.shape[1]:
            raise RuntimeError("Bluebild: A must be symmetrical.")

        if n_eig == 0:
            n_eig = A.shape[1]

        # make sure input is in coloumn major order
        if A.strides[0] != A.itemsize:
            A = np.array(A, order="F")
        if B is not None and B.strides[0] != B.itemsize:
            B = np.array(B, order="F")

        V = np.empty([A.shape[0], n_eig], order="F", dtype=A.dtype)
        D = np.empty(
            n_eig,
            order="F",
            dtype=np.float32 if A.dtype == np.complex64 else np.float64,
        )

        b_ptr = ctypes.c_void_p(0)
        ldb = 0
        if B is not None:
            b_ptr = B.ctypes.data
            ldb = int(B.strides[1] / B.itemsize)

        n_eig_out = ctypes.c_int(0)
        if A.dtype == np.complex64:
            _bluebild_eigh_s(
                self.ctx,
                A.shape[0],
                n_eig,
                A.ctypes.data,
                int(A.strides[1] / A.itemsize),
                b_ptr,
                ldb,
                ctypes.byref(n_eig_out),
                D.ctypes.data,
                V.ctypes.data,
                int(V.strides[1] / V.itemsize),
            )
        elif A.dtype == np.complex128:
            _bluebild_eigh_d(
                self.ctx,
                A.shape[0],
                n_eig,
                A.ctypes.data,
                int(A.strides[1] / A.itemsize),
                b_ptr,
                ldb,
                ctypes.byref(n_eig_out),
                D.ctypes.data,
                V.ctypes.data,
                int(V.strides[1] / V.itemsize),
            )
        else:
            raise TypeError("Bluebild: Input type must be complex.")

        return n_eig_out, D, V

    def sensitivity_field_data(self, n_eig, XYZ, W, wl):
        n_antenna = XYZ.shape[0]
        n_beam = W.shape[1]

        V = np.empty([n_beam, n_eig], order="F", dtype=W.dtype)
        D = np.empty(
            n_eig,
            order="F",
            dtype=np.float32 if W.dtype == np.complex64 else np.float64,
        )

        # make sure input is in coloumn major order
        if XYZ.strides[0] != XYZ.itemsize:
            XYZ = np.array(XYZ, order="F")
        if W.strides[0] != W.itemsize:
            W = np.array(W, order="F")

        if W.dtype == np.complex64:
            ier = _bluebild_sensitivity_field_data_s(
                self.ctx,
                wl,
                n_antenna,
                n_beam,
                n_eig,
                W.ctypes.data,
                int(W.strides[1] / W.itemsize),
                XYZ.ctypes.data,
                int(XYZ.strides[1] / XYZ.itemsize),
                D.ctypes.data,
                V.ctypes.data,
                int(V.strides[1] / V.itemsize),
            )
        elif W.dtype == np.complex128:
            ier = _bluebild_sensitivity_field_data_d(
                self.ctx,
                wl,
                n_antenna,
                n_beam,
                n_eig,
                W.ctypes.data,
                int(W.strides[1] / W.itemsize),
                XYZ.ctypes.data,
                int(XYZ.strides[1] / XYZ.itemsize),
                D.ctypes.data,
                V.ctypes.data,
                int(V.strides[1] / V.itemsize),
            )
        else:
            raise TypeError("Bluebild: Input type must be complex.")

        if ier != 0:
            raise RuntimeError("Bluebild: Failed to execute.")
        return D, V

    def intensity_field_data(self, n_eig, XYZ, W, wl, S, centroids):
        n_antenna = XYZ.shape[0]
        n_beam = W.shape[1]

        V = np.empty([S.shape[0], n_eig], order="F", dtype=S.dtype)
        D = np.empty(
            n_eig,
            order="F",
            dtype=np.float32 if S.dtype == np.complex64 else np.float64,
        )

        # make sure input is in coloumn major order
        if S.strides[0] != S.itemsize:
            S = np.array(S, order="F")
        if XYZ.strides[0] != XYZ.itemsize:
            XYZ = np.array(XYZ, order="F")
        if W.strides[0] != W.itemsize:
            W = np.array(W, order="F")
        if centroids.strides[0] != centroids.itemsize:
            centroids = np.array(centroids, order="F")

        cluster_idx = np.empty(n_eig, order="F", dtype=np.intc)

        if S.dtype == np.complex64:
            ier = _bluebild_intensity_field_data_s(
                self.ctx,
                wl,
                n_antenna,
                n_beam,
                n_eig,
                S.ctypes.data,
                int(S.strides[1] / S.itemsize),
                W.ctypes.data,
                int(W.strides[1] / W.itemsize),
                XYZ.ctypes.data,
                int(XYZ.strides[1] / XYZ.itemsize),
                D.ctypes.data,
                V.ctypes.data,
                int(V.strides[1] / V.itemsize),
                len(centroids),
                centroids.ctypes.data,
                cluster_idx.ctypes.data,
            )
        elif S.dtype == np.complex128:
            ier = _bluebild_intensity_field_data_d(
                self.ctx,
                wl,
                n_antenna,
                n_beam,
                n_eig,
                S.ctypes.data,
                int(S.strides[1] / S.itemsize),
                W.ctypes.data,
                int(W.strides[1] / W.itemsize),
                XYZ.ctypes.data,
                int(XYZ.strides[1] / XYZ.itemsize),
                D.ctypes.data,
                V.ctypes.data,
                int(V.strides[1] / V.itemsize),
                len(centroids),
                centroids.ctypes.data,
                cluster_idx.ctypes.data,
            )
        else:
            raise TypeError("Bluebild: Input type must be complex.")

        if ier != 0:
            raise RuntimeError("Bluebild: Failed to execute.")

        return D, V, cluster_idx

    def gram_matrix(self, XYZ, W, wl):
        N_antenna = XYZ.shape[0]
        N_beam = W.shape[1]
        G = np.empty(
            [N_beam, N_beam],
            order="F",
            dtype=np.complex64 if XYZ.dtype == np.float32 else np.complex128,
        )

        # make sure input is in coloumn major order
        if XYZ.strides[0] != XYZ.itemsize:
            XYZ = np.array(XYZ, order="F")
        if W.strides[0] != W.itemsize:
            W = np.array(W, order="F")

        if XYZ.dtype == np.float32:
            wl = np.float32(wl)
            ier = _bluebild_gram_matrix_s(
                self.ctx,
                N_antenna,
                N_beam,
                W.ctypes.data,
                int(W.strides[1] / W.itemsize),
                XYZ.ctypes.data,
                int(XYZ.strides[1] / XYZ.itemsize),
                wl,
                G.ctypes.data,
                int(G.strides[1] / G.itemsize),
            )
        elif XYZ.dtype == np.float64:
            wl = np.float64(wl)
            ier = _bluebild_gram_matrix_d(
                self.ctx,
                N_antenna,
                N_beam,
                W.ctypes.data,
                int(W.strides[1] / W.itemsize),
                XYZ.ctypes.data,
                int(XYZ.strides[1] / XYZ.itemsize),
                wl,
                G.ctypes.data,
                int(G.strides[1] / G.itemsize),
            )
        else:
            raise TypeError("Bluebild: Input type must be scalar.")

        if ier != 0:
            raise RuntimeError("Bluebild: Failed to execute.")
        return G



class Nufft3d3:
    def __init__(self, iflag, tol, num_trans, x, y, z, s, t, u, ctx=Context()):
        self.plan = ctypes.c_void_p(None)
        if x.dtype == np.float32:
            self.dtype = np.float32
        else:
            self.dtype = np.float64

        self.M = x.shape[0]
        self.N = s.shape[0]
        self.num_trans = num_trans

        if x.strides[0] != x.itemsize or x.dtype != self.dtype:
            x = np.array(x, dtype=self.dtype)
        if y.strides[0] != y.itemsize or y.dtype != self.dtype:
            y = np.array(y, dtype=self.dtype)
        if z.strides[0] != z.itemsize or z.dtype != self.dtype:
            z = np.array(z, dtype=self.dtype)
        if s.strides[0] != s.itemsize or s.dtype != self.dtype:
            s = np.array(s, dtype=self.dtype)
        if t.strides[0] != t.itemsize or t.dtype != self.dtype:
            t = np.array(t, dtype=self.dtype)
        if u.strides[0] != u.itemsize or u.dtype != self.dtype:
            u = np.array(u, dtype=self.dtype)

        if self.dtype == np.float32:
            ier = _bluebild_nufft3d3_create_s(ctx.ctx, iflag, tol, num_trans, self.M, x.ctypes.data,
                    y.ctypes.data, z.ctypes.data, self.N, s.ctypes.data, t.ctypes.data, u.ctypes.data, ctypes.byref(self.plan))
        else:
            ier = _bluebild_nufft3d3_create_d(ctx.ctx, iflag, tol, num_trans, self.M, x.ctypes.data,
                    y.ctypes.data, z.ctypes.data, self.N, s.ctypes.data, t.ctypes.data, u.ctypes.data, ctypes.byref(self.plan))
        if ier != 0:
            raise RuntimeError("Bluebild: Failed to create nufft3d3 plan.")

    def __del__(self):
        if self.plan:
            if self.dtype == np.float32:
                ier = _bluebild_nufft3d3_destroy_s(ctypes.byref(self.plan))
            else:
                ier = _bluebild_nufft3d3_destroy_d(ctypes.byref(self.plan))

            if ier != 0:
                raise RuntimeError("Bluebild: Failed to destroy nufft3d3 plan.")

    def execute(self, cj):
        c_dtype = np.complex64 if self.dtype == np.float32 else np.complex128
        fk = np.empty(self.N * self.num_trans, dtype=c_dtype)

        if cj.dtype != c_dtype or cj.strides[0] != cj.itemsize:
            cj = np.array(cj, dtype=c_dtype, order='F')

        if c_dtype == np.complex64:
            ier = _bluebild_nufft3d3_execute_s(self.plan, cj.ctypes.data, fk.ctypes.data)
        else:
            ier = _bluebild_nufft3d3_execute_d(self.plan, cj.ctypes.data, fk.ctypes.data)

        if ier != 0:
            raise RuntimeError("Bluebild: Failed to execute nufft3d3 plan.")
        return fk

