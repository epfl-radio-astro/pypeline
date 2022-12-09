#numactl --physcpubind=0 python test_parallel_synthesis.py

import os
if os.getenv('OMP_NUM_THREADS') == None : os.environ['OMP_NUM_THREADS'] = "1"

import bluebild_tools.cupy_util as bbt_cupy
use_cupy = bbt_cupy.is_cupy_usable()

import numpy as np
import numexpr as ne
from numpy import matmul
from scipy.linalg import expm
import time
import scipy.constants as constants
import scipy.linalg as linalg

from numpy.ctypeslib import ndpointer
from ctypes import *
from data_gen_utils import RandomDataGen

#local imports
import timing


# For CuPy agnostic code
# ----------------------
xp = bbt_cupy.cupy if use_cupy else np


def make_complexdouble_array(c):
    return np.array([c]).astype(np.complex128)


#################################################################################
# Custom MM
#################################################################################
def zgemm(A,B, a = 1, b = 0):
    # setting up C function
    so_file = "/home/etolley/bluebild/pypeline/src/zgemm-splat.so"
    custom_functions = CDLL(so_file)
    custom_functions.zgemm.argtypes=[c_int, c_int, c_int,
                                    ndpointer(dtype=np.complex128,ndim=1,flags='C'), #alpha
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.complex128,ndim=1,flags='C'), # beta
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int]
    #prep zgemm.c inputs
    (M,K) = A.shape
    (K,N) = B.shape
    A = A.astype(np.complex128, order = 'F')
    B = B.astype(np.complex128, order = 'F')
    ldA = M
    ldB = K
    C = np.zeros( (M,N),dtype =np.complex128, order='F')
    ldC = M
    alpha = make_complexdouble_array(a)
    beta =  make_complexdouble_array(b)


    # function call
    custom_functions.zgemm(M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC)

    return C

def zgemmexp(A,B, a , b = 0):
    # setting up C function
    so_file = "/home/etolley/bluebild/pypeline/src/zgemm-splat.so"
    custom_functions = CDLL(so_file)
    custom_functions.zgemmexp.argtypes=[c_int, c_int, c_int,
                                    ndpointer(dtype=np.complex128,ndim=1,flags='C'), #alpha
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int]
    #prep zgemm.c inputs
    (M,K) = A.shape
    (K,N) = B.shape
    A = A.astype(np.complex128, order = 'F')
    B = B.astype(np.complex128, order = 'F')
    ldA = M
    ldB = K
    C = np.zeros( (M,N),dtype =np.complex128, order='F')
    ldC = M
    alpha = make_complexdouble_array(a)


    # function call
    custom_functions.zgemmexp(M, N, K, alpha, A, ldA, B, ldB, C, ldC)

    return C

def dgemm(A,B, a = 1, b = 0):
    t0 = time.process_time() 
    # setting up C function
    so_file = "/home/etolley/bluebild/pypeline/src/dgemm-simple.so"
    custom_functions = CDLL(so_file)
    custom_functions.dgemm.argtypes=[c_int, c_int, c_int,
                                    c_double, #alpha
                                    ndpointer(dtype=np.float64,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.float64,ndim=2,flags='F'), c_int,
                                    c_double, # beta
                                    ndpointer(dtype=np.float64,ndim=2,flags='F'), c_int]
    #prep zgemm.c inputs
    (M,K) = A.shape
    (K,N) = B.shape
    A = np.asfortranarray(A)
    B = np.asfortranarray(B)
    ldA = M
    ldB = K
    C = np.zeros( (M,N),dtype =np.float64, order='F')
    ldC = M

    # function call
    t1 = time.process_time() 
    custom_functions.dgemm(M, N, K, a, A, ldA, B, ldB, b, C, ldC)

    return C

def dgemmexp(A,B, a = 1):
    # setting up C function
    so_file = "/home/etolley/bluebild/pypeline/src/dgemm-simple.so"
    custom_functions = CDLL(so_file)
    custom_functions.dgemmexp.argtypes=[c_int, c_int, c_int,
                                    c_double, #alpha
                                    ndpointer(dtype=np.float64,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.float64,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int]
    #prep zgemm.c inputs
    (M,K) = A.shape
    (K,N) = B.shape
    print(type(A))
    A = np.asfortranarray(A)
    B = np.asfortranarray(B)
    ldA = M
    ldB = K
    C = np.zeros( (M,N),dtype =np.complex128, order='F')
    ldC = M

    # function call
    custom_functions.dgemmexp(M, N, K, a, A, ldA, B, ldB, C, ldC)

    return C
#################################################################################
# Synthesis kernel
#################################################################################
def synthesize(pixGrid, V, XYZ, W, wl):    

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    print("Tensordot input shapes:", XYZ.shape, pixGrid.shape)

    a = 1j * 2 * np.pi / wl
    B = np.tensordot(XYZ, pixGrid, axes=1)
    P = np.zeros(B.shape,dtype=np.complex64)
    ne.evaluate( "exp(A * B)",dict(A=a, B=B),out=P,casting="same_kind",) 
    # P has shape (N_antenna,  N_height, N_width)

    print("Tensordot input shapes:", W.T.shape, P.shape)
    PW = np.tensordot(W.T, P, axes=1)
    # PW has shape (N_beam, N_height, N_width)

    print("Tensordot input shapes:", V.T.shape, PW.shape)
    E = np.tensordot(V.T, PW, axes=1)
    I = E.real ** 2 + E.imag ** 2
    # I has shape (N_eig, Nheight, N_width)

    return I

def synthesize_loop(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 
    N_eig = V.shape[1]

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    a = 1j * 2 * np.pi / wl
    E  = np.zeros((N_eig, N_height, N_width),   dtype=np.complex64)

    for i in range(N_width): #iterate over N_width
        #print("On iteration {0} of {1}".format(i, N_width))
        B  = matmul(XYZ, pixGrid[:,:,i])
        P = np.zeros(B.shape,dtype=np.complex64)
        ne.evaluate( "exp(A * B)",dict(A=a, B=B),out=P,casting="same_kind",)         
        PW = matmul(W.T,P)
        E[:,:,i]  = matmul(V.T, PW)
    I = E.real ** 2 + E.imag ** 2

    return I

def synthesize_loop_gpu(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 
    N_eig = V.shape[1]

    print(W.dtype, W.shape, W)

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)
    XYZ_gpu = xp.asarray(XYZ)
    WT_gpu = xp.asarray(W.T)
    VT_gpu = xp.asarray(V.T)

    a = 1j * 2 * np.pi / wl
    E  = np.zeros((N_eig, N_height, N_width),   dtype=np.complex64)

    for i in range(N_width): #iterate over N_width
        #print("On iteration {0} of {1}".format(i, N_width))
        pix_gpu = xp.asarray(pixGrid[:,:,i])

        B  = xp.matmul(XYZ_gpu, pix_gpu)
        P  = xp.exp(B*a)        
        PW = xp.matmul(WT_gpu,P)
        E_part = xp.matmul(VT_gpu, PW)

        E[:,:,i] = E_part.get()
    I = E.real ** 2 + E.imag ** 2

    return I

# does the same calculations as synthesize but reshaping
# the matrix stack instead of uisng tensordot. 
'''def synthesize_reshape(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 
    N_eig = V.shape[1]

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    a = 1j * 2 * np.pi / wl

    #flatten the height and width
    pixGrid = pixGrid.reshape(pixGrid.shape[0], N_height * N_width)
    t0 = time.process_time()
    B = XYZ @ pixGrid

    P = np.zeros(B.shape,dtype=np.complex64)
    ne.evaluate( "exp(A * B)",dict(A=a, B=B),out=P,casting="same_kind",) 
    #P = np.exp(a*B) # introduces some differences at fp level
    PW = W.T @ P
    E  = V.T @ PW
    I  = E.real ** 2 + E.imag ** 2
    I  = I.reshape(I.shape[0],N_height, N_width)

    return I'''

'''def synthesize_test(pixGrid, V, XYZ, W, wl): 
    print("Test reshape") 
    N_eig = V.shape[1]
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 

    

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)
    E = np.zeros( (N_eig,N_height, N_width) )
    a = 1j * 2 * np.pi / wl

    print("Tensordot input shapes:", XYZ.shape, pixGrid.shape)

    for w in range(N_width):
        B = matmul(XYZ, pixGrid[:,:,w])
        P = np.zeros(B.shape,dtype=np.complex64)
        ne.evaluate( "exp(A * B)",dict(A=a, B=B),out=P,casting="same_kind",) 
        # P has shape (N_antenna,  N_height, N_width)

        PW = np.tensordot(W.T, P, axes=1)
        # PW has shape (N_beam, N_height, N_width)

        E[:,:,w] = np.tensordot(V.T, PW, axes=1)

    I = E.real ** 2 + E.imag ** 2
    print(I.shape, N_eig,N_height, N_width )
    # I has shape (N_eig, Nheight, N_width)

    return I'''

def zgemm_synthesize_reshape(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 
    N_eig = V.shape[1]

    #V = np.asfortranarray(V)
    #W = np.asfortranarray(W)

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    a = 1j * 2 * np.pi / wl

    #flatten the height and width
    pixGrid = pixGrid.reshape(pixGrid.shape[0], N_height * N_width)

    t1 = time.process_time()
    B = zgemm(XYZ,pixGrid)

    P = np.zeros(B.shape,dtype=np.complex64)
    ne.evaluate( "exp(A * B)",dict(A=a, B=B),out=P,casting="same_kind",) 

    PW = W.T @ P
    E  = V.T @ PW
    I  = E.real ** 2 + E.imag ** 2
    I  = I.reshape(I.shape[0],N_height, N_width)

    return I

def dgemm_synthesize_reshape(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 
    N_eig = V.shape[1]

    #V = np.asfortranarray(V)
    #W = np.asfortranarray(W)

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    a = 1j * 2 * np.pi / wl

    #flatten the height and width
    pixGrid = pixGrid.reshape(pixGrid.shape[0], N_height * N_width)

    t1 = time.process_time()
    B = dgemm(XYZ,pixGrid)

    P = np.zeros(B.shape,dtype=np.complex64)
    ne.evaluate( "exp(A * B)",dict(A=a, B=B),out=P,casting="same_kind",) 

    PW = W.T @ P
    E  = V.T @ PW
    I  = E.real ** 2 + E.imag ** 2
    I  = I.reshape(I.shape[0],N_height, N_width)

    return I


def zgemmexp_synthesize_reshape(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 
    N_eig = V.shape[1]

    #V = np.asfortranarray(V) 
    #W = np.asfortranarray(W)

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    # dgemmexp will apply the imaginary component to a
    a = 2 * np.pi / wl

    #flatten the height and width
    pixGrid = pixGrid.reshape(pixGrid.shape[0], N_height * N_width)

    P = zgemmexp(XYZ,pixGrid,a)

    PW = W.T @ P
    E  = V.T @ PW
    I  = E.real ** 2 + E.imag ** 2
    I  = I.reshape(I.shape[0],N_height, N_width)

    return I

def dgemmexp_synthesize_reshape(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 
    N_eig = V.shape[1]

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    # dgemmexp will apply the imaginary component to a
    a = 2 * np.pi / wl

    #flatten the height and width
    pixGrid = pixGrid.reshape(pixGrid.shape[0], N_height * N_width)

    P = dgemmexp(XYZ,pixGrid,a)

    PW = W.T @ P
    E  = V.T @ PW
    I  = E.real ** 2 + E.imag ** 2
    I  = I.reshape(I.shape[0],N_height, N_width)

    return I

#################################################################################

def test_dgemm(data, string = ""):
    A2 = data.getXYZ()
    B2 = data.getPixGrid()
    B2 = B2 / linalg.norm(B2, axis=0)
    A2 = A2 - A2.mean(axis=0)
    B2 = B2.reshape(B2.shape[0], B2.shape[1] * B2.shape[2])

    t2 = time.process_time()
    result = dgemm(A2,B2)
    print("DGEMM", string, "time", time.process_time() - t2)

if __name__ == "__main__":

    # parameters
    frequency = 145e6
    wl = constants.speed_of_light / frequency

    data = RandomDataGen(32, N_station = 24) # 24 or 37

    timer = timing.Timer()

    pix = data.getPixGrid()

    for t in range(0,1):
        (V, XYZ, W) = data.getVXYZW(t)

        print(W.dtype, V.dtype)

        # call an alternate dummy synthesis kernel which reshapes the matrices

        timer.start_time("Test dummy synthesis")
        stat_sdum = synthesize_loop(pix,V,XYZ,W, wl)
        timer.end_time("Test dummy synthesis")

         # call an alternate dummy synthesis kernel which reshapes the matrices
        if use_cupy:
            timer.start_time("GPU dummy synthesis")
            stat_gpu = synthesize_loop_gpu(pix,V,XYZ,W, wl)
            timer.end_time("GPU dummy synthesis")

        # call the dummy synthesis kernal
        timer.start_time("Dummy synthesis")
        stat_dum  = synthesize(pix,V,XYZ,W, wl)
        timer.end_time("Dummy synthesis")

        #
        ##TODO: remove paths to Emma's private space
        #
        """
        # call an alternate dummy synthesis kernel which uses a special DGEMM
        timer.start_time("DGEMM dummy synthesis")
        stat_gdum = dgemm_synthesize_reshape(pix,V,XYZ,W, wl)
        timer.end_time("DGEMM dummy synthesis")

        # call an alternate dummy synthesis kernel which uses an extra special ZGEMM
        timer.start_time("DGEMMexp dummy synthesis")
        stat_gexpdum =  dgemmexp_synthesize_reshape(pix,V,XYZ,W, wl)
        timer.end_time("DGEMMexp dummy synthesis")
        """

        print("Avg diff between dummy & test-dummy synthesizers:", np.average( stat_dum - stat_sdum))
        if use_cupy:
            print("Avg diff between dummy & gpu-based synthesizers:", np.average( stat_dum - stat_gpu))
        #print("Avg diff between dummy & DGEMM synthesizers:", np.max( np.abs(stat_dum - stat_gdum)))
        #print("Avg diff between dummy & DGEMMexp synthesizers:", np.max( np.abs(stat_dum - stat_gexpdum)))
        #print("Avg diff between DGEMM & DGEMMexp synthesizers:", np.max( np.abs(stat_gdum - stat_gexpdum)))

    print(timer.summary())




