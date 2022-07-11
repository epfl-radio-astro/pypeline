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

# EO: Don't use timesteps with large spacing as this will be used for the integration time
#     I will submit a PR when I get the rights to push to RASCIL repo
# ----------------------------------------------------------------------------------------
#times = (numpy.pi / 43200.0) * numpy.arange(-4 * 3600, +4 * 3600.0, 1800)
#times = numpy.array([0.0, 1.0*numpy.pi/12.0, 2.0*numpy.pi/12.0, 3.0*numpy.pi/12.0])
#times = numpy.array([0.0, 1.0*numpy.pi/12.0])
times = numpy.array([0.0])
print(f"Input times: {times}")

#frequency = numpy.linspace(1.0e8, 1.1e8, 5)
frequency = numpy.array([1.0e8])

#channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
channel_bandwidth = numpy.array([1.0e6])

# Define the component and give it some spectral behaviour
phasecentre = SkyCoord(ra=20.0 * u.deg, dec=15.0 * u.deg, frame='icrs', equinox='J2000')

xvis = create_blockvisibility(lowcore, times, frequency,
                              channel_bandwidth=channel_bandwidth,
                              phasecentre=phasecentre,
                              integration_time=8.0,
                              polarisation_frame=PolarisationFrame("linear"),
                              weight=1.0,
                              times_are_ha=True)

NT = len(times)
NF = len(frequency)
ref_size = (NT, 13861, NF, 4)
assert xvis['vis'].shape == ref_size, xvis['vis'].shape
assert xvis["uvw"].data.shape == (NT, 13861, 3), xvis["uvw"].shape
assert xvis["uvw_lambda"].data.shape == (NT, 13861, NF, 3), xvis["uvw_lambda"].data.shape

plot_uvcoverage([xvis])
plt.savefig("%s/LOWBD2-CORE_uv_coverage"%results_dir)

advice = advise_wide_field(xvis, guard_band_image=3.0, delA=0.1, facets=1, wprojection_planes=1,
                           oversampling_synthesised_beam=4.0)
cellsize = advice['cellsize']
m31image = create_test_image(frequency=frequency, cellsize=cellsize, phasecentre=phasecentre,
                             channel_bandwidth=channel_bandwidth, polarisation_frame=PolarisationFrame("linear"))
#print(m31image)
#nchan, npol, ny, nx = m31image["pixels"].data.shape
#print("xvis in nchan, npol, ny, nx", nchan, npol, ny, nx)

fig=show_image(m31image)
plt.savefig("%s/test_image_true"%results_dir)

xvis = predict_ng(xvis, m31image, context='2d')
#print("xvis[uvw] =", xvis["uvw"])
plt.clf()
plot_visibility([xvis])
plt.savefig("%s/visibility"%results_dir)

export_blockvisibility_to_ms("%s/ska-pipeline_simulation.ms"%results_dir,[xvis])
