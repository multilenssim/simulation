#See: https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/
from __future__ import print_function, division
import numpy as np
import timeit  # useless - if the guy using a special interpreter or something in the blog post above?
from time import time
import numba

# Install nufft from http://github.com/dfm/python-nufft/
from nufft import nufft1 as nufft_fortran

#@jit # Seems to do nothing
def nufftfreqs(M, df=1):
    """Compute the frequency range used in nufft for M frequency bins"""
    return df * np.arange(-(M // 2), M - (M // 2))


#@jit # Seems to do nothing
def nudft(x, y, M, df=1.0, iflag=1):
    """Non-Uniform Direct Fourier Transform"""
    sign = -1 if iflag < 0 else 1
    return (1 / len(x)) * np.dot(y, np.exp(sign * 1j * nufftfreqs(M, df) * x[:, np.newaxis]))

def _compute_grid_params(M, eps):
    # Choose Msp & tau from eps following Dutt & Rokhlin (1993)
    if eps <= 1E-33 or eps >= 1E-1:
        raise ValueError("eps = {0:.0e}; must satisfy 1e-33 < eps < 1e-1.".format(eps))
    ratio = 2 if eps > 1E-11 else 3
    Msp = int(-np.log(eps) / (np.pi * (ratio - 1) / (ratio - 0.5)) + 0.5)
    Mr = max(ratio * M, 2 * Msp)
    lambda_ = Msp / (ratio * (ratio - 0.5))
    tau = np.pi * lambda_ / M ** 2
    return Msp, Mr, tau

# @numba.jit # Factor of 10x improvement                                                           
def nufft_python(x, c, M, df=1.0, eps=1E-15, iflag=1):
    """Fast Non-Uniform Fourier Transform with Python"""
    Msp, Mr, tau = _compute_grid_params(M, eps)
    N = len(x)
    
    # Construct the convolved grid
    ftau = np.zeros(Mr, dtype=c.dtype)
    Mr = ftau.shape[0]
    hx = 2 * np.pi / Mr
    mm = np.arange(-Msp, Msp)
    for i in range(N):
        xi = (x[i] * df) % (2 * np.pi)
        m = 1 + int(xi // hx)
        spread = np.exp(-0.25 * (xi - hx * (m + mm)) ** 2 / tau)
        ftau[(m + mm) % Mr] += c[i] * spread
        
    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / Mr) * np.fft.fft(ftau)
    else:
        Ftau = np.fft.ifft(ftau)
    Ftau = np.concatenate([Ftau[-(M//2):], Ftau[:M//2 + M % 2]])
            
    # Deconvolve the grid using convolution theorem
    k = nufftfreqs(M)
    return (1 / N) * np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau

# nopython=True means an error will be raised
# if fast compilation is not possible.
@numba.jit(nopython=True)
def build_grid(x, c, tau, Msp, ftau):
    Mr = ftau.shape[0]
    hx = 2 * np.pi / Mr
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi)
        m = 1 + int(xi // hx)
        for mm in range(-Msp, Msp):
            ftau[(m + mm) % Mr] += c[i] * np.exp(-0.25 * (xi - hx * (m + mm)) ** 2 / tau)
    return ftau

def nufft_numba(x, c, M, df=1.0, eps=1E-15, iflag=1):
    """Fast Non-Uniform Fourier Transform with Numba"""
    Msp, Mr, tau = _compute_grid_params(M, eps)
    N = len(x)
    
    # Construct the convolved grid
    ftau = build_grid(x * df, c, tau, Msp, np.zeros(Mr, dtype=c.dtype))
    
    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / Mr) * np.fft.fft(ftau)
    else:
        Ftau = np.fft.ifft(ftau)
    Ftau = np.concatenate([Ftau[-(M//2):], Ftau[:M//2 + M % 2]])
        
    # Deconvolve the grid using convolution theorem
    k = nufftfreqs(M)
    return (1 / N) * np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau
                                                        
def nufft_numpy(x, y, M, df=1.0, iflag=1, eps=1E-15):
    """Fast Non-Uniform Fourier Transform"""
    Msp, Mr, tau = _compute_grid_params(M, eps)
    N = len(x)
    
    # Construct the convolved grid ftau:
    # this replaces the loop used above
    ftau = np.zeros(Mr, dtype=y.dtype)
    hx = 2 * np.pi / Mr
    xmod = (x * df) % (2 * np.pi)
    m = 1 + (xmod // hx).astype(int)
    mm = np.arange(-Msp, Msp)
    mpmm = m + mm[:, np.newaxis]
    spread = y * np.exp(-0.25 * (xmod - hx * mpmm) ** 2 / tau)
    np.add.at(ftau, mpmm % Mr, spread)
    
    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / Mr) * np.fft.fft(ftau)
    else:
        Ftau = np.fft.ifft(ftau)
    Ftau = np.concatenate([Ftau[-(M//2):], Ftau[:M//2 + M % 2]])
        
    # Deconvolve the grid using convolution theorem
    k = nufftfreqs(M)
    return (1 / N) * np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau

@numba.jit(nopython=True)
def build_grid_fast(x, c, tau, Msp, ftau, E3):
    Mr = ftau.shape[0]
    hx = 2 * np.pi / Mr
    
    # precompute some exponents
    for j in range(Msp + 1):
        E3[j] = np.exp(-(np.pi * j / Mr) ** 2 / tau)

    # spread values onto ftau
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi)
        m = 1 + int(xi // hx)
        xi = (xi - hx * m)
        E1 = np.exp(-0.25 * xi ** 2 / tau)
        E2 = np.exp((xi * np.pi) / (Mr * tau))
        E2mm = 1
        for mm in range(Msp):
            ftau[(m + mm) % Mr] += c[i] * E1 * E2mm * E3[mm]
            E2mm *= E2
            ftau[(m - mm - 1) % Mr] += c[i] * E1 / E2mm * E3[mm + 1]
    return ftau
            
            
def nufft_numba_fast(x, c, M, df=1.0, eps=1E-15, iflag=1):
    """Fast Non-Uniform Fourier Transform with Numba"""
    Msp, Mr, tau = _compute_grid_params(M, eps)
    N = len(x)
    
    # Construct the convolved grid
    ftau = build_grid_fast(x * df, c, tau, Msp, np.zeros(Mr, dtype=c.dtype), np.zeros(Msp + 1, dtype=x.dtype))
    
    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / Mr) * np.fft.fft(ftau)
    else:
        Ftau = np.fft.ifft(ftau)
    Ftau = np.concatenate([Ftau[-(M//2):], Ftau[:M//2 + M % 2]])
    
    # Deconvolve the grid using convolution theorem
    k = nufftfreqs(M)
    return (1 / N) * np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau


def test_nufft(nufft_func, M=1000, Mtime=100000):
    # Test vs the direct method
    print(30 * '-')
    name = {'nufft1':'nufft_fortran'}.get(nufft_func.__name__, nufft_func.__name__)
    print("testing {0}".format(name))
    rng = np.random.RandomState(0)
    x = 100 * rng.rand(M + 1)
    y = np.sin(x)
    for df in [1, 2.0]:
        for iflag in [1, -1]:
            F1 = nudft(x, y, M, df=df, iflag=iflag)
            F2 = nufft_func(x, y, M, df=df, iflag=iflag)
            assert np.allclose(F1, F2)
    print("- Results match the DFT")
            
    # Time the nufft function
    x = 100 * rng.rand(Mtime)
    y = np.sin(x)
    times = []
    for i in range(5):
        t0 = time()
        F = nufft_func(x, y, Mtime)
        t1 = time()
        times.append(t1 - t0)
        print("- Execution time (M={0}): {1:.2g} sec".format(Mtime, np.median(times)))
                                                                                                                                                    

test_nufft(nufft_fortran)
#test_nufft(nudft)  Causes "memory eror"
test_nufft(nufft_numpy)
test_nufft(nufft_numba)
test_nufft(nufft_numba_fast)
test_nufft(nufft_python)

'''
x = 100 * np.random.random(1000)
y = np.sin(x)

start = time.time()
Y1 = nudft(x, y, 100000)
print(time.time() - start)


start = time.time()
Y2 = nufft_fortran(x, y, 100000)
print(time.time() - start)

print(np.shape(Y1))
print(np.shape(Y2))

print(np.allclose(Y1, Y2))
'''
#print(timeit.timeit("nufft_fortran(x, y, 1000)", setup="from __main__ import nudft, x, y"))
#print(timeit.timeit("nufft_fortran(x, y, 1000)", setup="from nufft import nufft1 as nufft_fortran"))
