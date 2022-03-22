# Using git-lfs for storing SKA example datasets

```
$ module load git git-lfs
```

```
$ git lfs track "datasets/**"
Tracking "datasets/**"
```

## Testing

$ ssh izar ...
$ Sinteract ...
$ conda activate pype-111
$ module load gcc cuda/11.0.2 fftw
$ cd pypeline
$ python benchmarking/ ...

## Example
```
import os
from pathlib import Path
datasets_dir = Path.joinpath(Path(__file__).absolute().parents[1], "datasets")
if not os.path.isdir(datasets_dir):
    print(f"Fatal  : datasets_dir {datasets_dir} not existing!")
ms_file = Path.joinpath(datasets_dir, "gauss4/gauss4_t201806301100_SBL180.MS").as_posix()
data = RealDataGen(Path.joinpath(datasets_dir, "gauss4/gauss4_t201806301100_SBL180.MS").as_posix(), N_level = 4, N_station = 37 ) # n level = # eigenimages
```

- [ ] benchmarking/read_bb_fits.py:30:with fits.open("/home/etolley/bluebild/pypeline/bluebild_ss_4gauss_37Stations.fits") as hdul:
- [ ] benchmarking/read_bb_fits.py:35:with fits.open("/home/etolley/casacore_setup/deconvScale-image.fits") as hdul: #
- [ ] benchmarking/read_bb_fits.py:39:with fits.open("/home/etolley//data/gauss4/gauss4-image-pb.fits") as hdul: #"/home/etolley/casacore_setup/deconv-image.fits"
- [ ] benchmarking/read_bb_fits.py:43:with fits.open("/home/etolley/data/gauss4- [x] benchmarking/generate_bb_output.py:177:    with fits.open("/home/etolley/data/gauss4/C_4gaussian-model.fits") as hdul:

- [x] benchmarking/test_fastsynthesizer.py:110:    data = RealDataGen("/work/scitas-share/SKA/data/gauss4/gauss4_t201806301100_SBL180.MS", N_level = 4, N_station = 24) # n level = # eigenimages

- [r] python/lofar_bootes_ps.py:29:ms_file = "/home/sep/Documents/Data/Radio-Astronomy/LOFAR/BOOTES24_SB180-189.2ch8s_SIM.ms"
- [r] python/lofar_bootes_ps.py:30:ms_file = "/home/etolley/data/RX42_SB100-109.2ch10s.ms"
- [r] python/lofar_bootes_ps.py:31:ms_file = "/home/etolley/data/claudio_inputs/test1hr_t201806301100_SBL150.MS"
- [x] python/lofar_bootes_ps.py:32:ms_file = "/home/etolley/data/gauss4/gauss4_t201806301100_SBL180.MS"

- [ ] python/meerkat.py:21:ms_file = "/scratch/foureste/Meerkat/1524929477.ms"

- [r] python/lofar_bootes_ss.py:30:ms_file = "/home/sep/Documents/Data/Radio-Astronomy/LOFAR/BOOTES24_SB180-189.2ch8s_SIM.ms"
- [r] python/lofar_bootes_ss.py:31:ms_file = "/home/etolley/data/RX42_SB100-109.2ch10s.ms"
- [r] python/lofar_bootes_ss.py:32:ms_file = "/home/etolley/data/gauss_emma/gauss1hr_t201806301100_SBL150.MS"
- [x] python/lofar_bootes_ss.py:33:ms_file = "/home/etolley/data/gauss4/gauss4_t201806301100_SBL180.MS"

- [x] python/examine_ms_file.py:27:ms_files = ["/home/etolley/data/gauss4/gauss4_t201806301100_SBL180.MS"]

- [x] examples/simulation/lofar_4gaus_ps.py:36:ms_file = "/home/etolley/data/gauss4/gauss4_t201806301100_SBL180.MS"
- [x] examples/simulation/lofar_4gaus_ps.py:149:cl_WCS = ifits.wcs("/home/etolley/data/gauss4/gauss4-image-pb.fits")

- [x] examples/simulation/lofar_4gaus_ss.py:41:ms_file = "/home/etolley/data/gauss4/gauss4_t201806301100_SBL180.MS"
- [x] examples/simulation/lofar_4gaus_ss.py:180:cl_WCS = ifits.wcs("/home/etolley/data/gauss4/gauss4-image-pb.fits")

- [ ] examples/simulation/lofar_toothbrush_ps.py:36:ms_file = "/Users/mmjasime/Documents/Datasets/RX42_SB100-109.2ch10s.ms"
- [ ] examples/simulation/lofar_toothbrush_ps.py:37:ms_file = "/home/etolley/data/toothbrush/RX42_SB100-109.2ch10s.ms"
- [ ] examples/simulation/lofar_toothbrush_ps.py:139:cl_WCS = ifits.wcs('/home/etolley//data/toothbrush/toothbrush-image.fits')

- [ ] examples/simulation/lofar_toothbrush_ss.py:35:ms_file = "/Users/mmjasime/Documents/Datasets/RX42_SB100-109.2ch10s.ms"
- [ ] examples/simulation/lofar_toothbrush_ss.py:36:ms_file = "/home/etolley/data/toothbrush/RX42_SB100-109.2ch10s.ms"
- [ ] examples/simulation/lofar_toothbrush_ss.py:138:cl_WCS = ifits.wcs('/home/etolley//data/toothbrush/toothbrush-image.fits')

- [ ] examples/real_data/lofar_bootes_ps.py:29:ms_file = "/home/sep/Documents/Data/Radio-Astronomy/LOFAR/BOOTES24_SB180-189.2ch8s_SIM.ms"

- [ ] examples/real_data/lofar_bootes_ss.py:30:ms_file = "/home/sep/Documents/Data/Radio-Astronomy/LOFAR/BOOTES24_SB180-189.2ch8s_SIM.ms"

- [x] jenkins/generic_synthesizer.py:120:    MS_file = "/work/scitas-share/SKA/data/gauss4/gauss4_t201806301100_SBL180.MS"

