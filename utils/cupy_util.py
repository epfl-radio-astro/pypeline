import os
import sys

def has_lmod():
    HAS_LMOD = os.environ.get('LMOD_CMD')
    #print("HAS_LMOD?", HAS_LMOD)
    if HAS_LMOD == None:
        return False
    else:
        return True

def is_cupy_available() -> bool:
    try:
        import cupy
        globals()['cupy'] = __import__('cupy')
        return True
    except ModuleNotFoundError as error:
        print("mnfe:", error)
        print("\n================================================================")
        print("Hint: you need to install CuPy")
        print("      see https://docs.cupy.dev/en/stable/install.html")
        print("================================================================\n")
        return False

    except ImportError as error:
        print("ie:", error)
        print("\n================================================================")
        print("Hint: did you load the correct version of cuda?")
        if has_lmod():
            cmd = 'module list'
            stream = os.popen(cmd)
            cmdout = stream.read().strip()            
            print('$ ' + cmd)
            print(cmdout + "\n")
            cmd = 'module spider cuda'
            stream = os.popen(cmd)
            cmdout = stream.read().strip()            
            print('$ ' + cmd)
            print(cmdout)
        print("================================================================\n")
        return False

    except Exception as error:
        error_type = type(error)
        print(f"Exception of type {error_type} detected but not handled -- FIX ME")
        sys.exit(1)

    return True


"""
Note: this function assumes that CuPy is availale. See is_cupy_available().
"""
def is_a_gpu_device_available() -> bool:
    try:
        cupy.cuda.runtime.getDeviceCount()
        return True
    except Exception as error:
        print(f"Error: exception of type {type(error)} raised.")
        print(error)
        print("\n================================================================")
        print("Hint: Check that you have a machine or an allocation with at")
        print("      least one Nvidia device available")
        cmd = 'nvidia-smi'
        print(f"$ {cmd}")
        stream = os.popen(cmd)
        cmdout = stream.read().strip()
        print(cmdout)
        print("================================================================\n")
        return False


def is_cupy_usable() -> bool:

    MUST_CUPY = os.environ.get('MUST_CUPY')

    if MUST_CUPY != None:
        if MUST_CUPY != '-1' and MUST_CUPY != '0' and MUST_CUPY != '1':
            print("Error: when set, MUST_CUPY must either be '-1', '0' or '1'.")
            sys.exit(1)
        MUST_CUPY = int(MUST_CUPY)
    else:
        MUST_CUPY = 0

    use_cupy = True

    if MUST_CUPY == -1:

        print("Info: CuPy explicitely NOT requested => will run on CPU")
        use_cupy = False

    elif MUST_CUPY == 0:

        if is_cupy_available():
            if is_a_gpu_device_available():
                print("Info: CuPy not explicitly requested but available, together with a GPU device, so will use it")
            else:
                print("Info: CuPy not explicitly requested but available, but no GPU device is available, so cannot use it")
                use_cupy = False
        else:
            print("Info: CuPy not explicitly requested and unavailable, so will compute on CPU")
            use_cupy = False

    elif MUST_CUPY == 1:

        if not is_cupy_available():
            print("Error: CuPy explicitly requested but unavailable")
            sys.exit(1)
        else:
            if is_a_gpu_device_available():
                print("Info: CuPy explicitly requested and available, together with a GPU device, so will use it")
            else:
                print("Error: CuPy explicitly requested and available, but no GPU device available")
                sys.exit(1)

    else:

        print("Error. Unknown case -- FIX ME")
        sys.exit(1)

    if use_cupy = False:
        os.environ["CUPY_PYLOPS"] = "0"
        os.environ["CPUY_PYFFS"]  = "0"

    return use_cupy
