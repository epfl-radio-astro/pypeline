
#### Install conda
​
## you must respond to the interactive instructions following this command
## be sure to install conda into your home directory
#bash Miniconda3-latest-Linux-x86_64.sh 
​
export PATH=$HOME/miniconda3/bin/:$PATH
conda update -n base -c defaults conda
pip install --upgrade setuptools

#### Install bluebild
​
#git clone https://github.com/etolley/pypeline.git
cd pypeline/
​
​
##  create conda environment
conda create --name=pypeline       \
             --channel=defaults    \
             --channel=conda-forge \
             --file=conda_requirements.txt
source pypeline.sh --no_shell

#pip install pycsou

module load gcc/8.4.0-cuda
module load cuda/10.2.89
pip install cupy-cuda102

cd ..
​
git clone https://github.com/imagingofthings/pyFFS.git
cd pyFFS/
#git checkout v1.0
mkdir install
python3 setup.py develop --user
​
cd ..
​
git clone -b dev https://github.com/imagingofthings/ImoT_tools.git
cd ImoT_tools/
#git checkout v1.0
git checkout dev #EO: otherwise no fits.py under io/
python3 setup.py develop --user
​
cd ..
​
git clone https://github.com/matthieumeo/pycsou.git
pip install -e pycsou/
​
cd pypeline/
# you may need to modify this path for your own directory structure
export PYTHONPATH="$PYTHONPATH:$HOME/bluebild/pyFFS/install:$HOME/bluebild/ImoT_tools/install"
​
python3 setup.py develop --user
​
# you may need to modify this path for your own directory structure
export PYTHONPATH="$PYTHONPATH:$HOME/bluebild/pypeline/install"
​
pip install scikit-image
pip install pycsou
​
#python3 test.py                # Run test suite (optional, recommended)
# python3 setup.py build_sphinx  # Generate documentation (optional)
