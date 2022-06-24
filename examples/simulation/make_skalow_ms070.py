import sys
import os
import numpy
from matplotlib import pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from rascil.data_models import PolarisationFrame
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_blockvisibility
from rascil.processing_components import plot_uvcoverage, advise_wide_field, create_test_image, \
                                         show_image, export_blockvisibility_to_ms, predict_ng, \
                                         plot_visibility


results_dir = './rascil_sim/'
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

lowcore = create_named_configuration('LOWBD2-CORE', rmax=1000)
#times = (numpy.pi / 43200.0) * numpy.arange(-4 * 3600, +4 * 3600.0, 1800)
#print(f"times = {times} {times.shape}")
times = numpy.zeros([1])
times = numpy.array([numpy.pi / 3])
#frequency = numpy.linspace(1.0e8, 1.1e8, 5)
frequency = numpy.array([1.0e8])
#channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
channel_bandwidth = numpy.array([1.0e6])
# Define the component and give it some spectral behaviour
#phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
phasecentre = SkyCoord(ra=+193.0 * u.deg, dec=-67.0 * u.deg, frame='icrs', equinox='J2000')
xvis = create_blockvisibility(lowcore, times, frequency,
                              channel_bandwidth=channel_bandwidth,
                              phasecentre=phasecentre,
                              integration_time=8.0,
                              polarisation_frame=PolarisationFrame("linear"),
                              weight=1.0)
NT = len(times)
NF = len(frequency)
assert xvis['vis'].shape == (NT, 13861, NF, 4), xvis['vis'].shape
assert xvis["uvw"].data.shape == (NT, 13861, 3), xvis["uvw"].shape
assert xvis["uvw_lambda"].data.shape == (NT, 13861, NF, 3), xvis["uvw_lambda"].data.shape

print(xvis)

plot_uvcoverage([xvis])
plt.savefig("%s/LOWBD2-CORE_uv_coverage"%results_dir)

advice = advise_wide_field(xvis, guard_band_image=3.0, delA=0.1, facets=1, wprojection_planes=1,
                           oversampling_synthesised_beam=4.0)
cellsize = advice['cellsize']
m31image = create_test_image(frequency=frequency, cellsize=cellsize, phasecentre=phasecentre,
                             channel_bandwidth=channel_bandwidth, polarisation_frame=PolarisationFrame("linear"))
nchan, npol, ny, nx = m31image["pixels"].data.shape
print("xvis in nchan, npol, ny, nx", nchan, npol, ny, nx)
fig=show_image(m31image)
plt.savefig("%s/test_image_true"%results_dir)

xvis = predict_ng(xvis, m31image, context='2d')
plt.clf()
plot_visibility([xvis])
plt.savefig("%s/visibility"%results_dir)

export_blockvisibility_to_ms("%s/ska-pipeline_simulation.ms"%results_dir,[xvis])


sys.exit(0)


# followed from
# https://ska-telescope.gitlab.io/external/rascil/examples/notebooks/imaging.html

import os
import sys

sys.path.append(os.path.join('..', '..'))

results_dir = './results_large/'
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

from matplotlib import pylab

pylab.rcParams['figure.figsize'] = (8.0, 8.0)
pylab.rcParams['image.cmap'] = 'rainbow'

import numpy

from astropy.coordinates import SkyCoord
from astropy import units as u

from matplotlib import pyplot as plt


from rascil.data_models import PolarisationFrame

from rascil.processing_components import create_blockvisibility, export_blockvisibility_to_ms, \
    show_image, export_image_to_fits, \
    deconvolve_cube, restore_cube, create_named_configuration, create_test_image, \
    create_image_from_visibility, advise_wide_field, invert_ng, predict_ng, \
    plot_uvcoverage, plot_visibility

import logging

linebreak = '========================================================'

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pylab.rcParams['figure.figsize'] = (12.0, 12.0)
pylab.rcParams['image.cmap'] = 'rainbow'

'''
    For LOWBD2, setting rmax gives the following number of stations
    100.0       13
    300.0       94
    1000.0      251
    3000.0      314
    10000.0     398
    30000.0     476
    100000.0    512

'''
#lowr3 = create_named_configuration('LOWBD2', rmax = 100000.0)
lowr3 = create_named_configuration('LOW')

print(linebreak)
print(lowr3)

times = numpy.zeros([1])
#times = numpy.linspace(-numpy.pi / 3.0, numpy.pi / 3.0, 5)
frequency = numpy.array([1e8])
channel_bandwidth = numpy.array([1e6])
phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
vt = create_blockvisibility(lowr3, times, frequency, channel_bandwidth=channel_bandwidth,
#                       weight=1.0, phasecentre=phasecentre, polarisation_frame=PolarisationFrame('stokesI'))
                       weight=1.0, phasecentre=phasecentre, polarisation_frame=PolarisationFrame('linear'))

print(linebreak)
print(vt)

advice = advise_wide_field(vt, guard_band_image=3.0, delA=0.1, facets=1, wprojection_planes=1,
                           oversampling_synthesised_beam=4.0)
cellsize = advice['cellsize']

plt.clf()
plot_uvcoverage([vt])
plt.savefig("%s/LOWBD2_uvcoverage"%results_dir)


#=============================================



m31image = create_test_image(frequency=frequency, cellsize=cellsize,
                             phasecentre=phasecentre)
nchan, npol, ny, nx = m31image["pixels"].data.shape

fig=show_image(m31image)
plt.savefig("%s/test_image_true"%results_dir)

#=============================================
vt = predict_ng(vt, m31image, context='2d')

plt.clf()
plot_visibility([vt])
plt.savefig("%s/visibility"%results_dir)

export_blockvisibility_to_ms("%s/ska-pipeline_simulation.ms"%results_dir,[vt])

#=============================================
model = create_image_from_visibility(vt, cellsize=cellsize, npixel=512)
dirty, sumwt = invert_ng(vt, model, context='2d')
psf, sumwt = invert_ng(vt, model, context='2d', dopsf=True)

show_image(dirty)
plt.savefig("%s/test_image_dirty"%results_dir)
print("Max, min in dirty image = %.6f, %.6f, sumwt = %f" % (dirty["pixels"].data.max(), dirty["pixels"].data.min(), sumwt))

print("Max, min in PSF         = %.6f, %.6f, sumwt = %f" % (psf["pixels"].data.max(), psf["pixels"].data.min(), sumwt))

export_image_to_fits(dirty, '%s/imaging_dirty.fits'%(results_dir))
export_image_to_fits(psf, '%s/imaging_psf.fits'%(results_dir))
