import sys
from spectral_cube import SpectralCube
import tqdm
import matplotlib.pyplot as plt
cube = SpectralCube.read(sys.argv[1])
# cube = SpectralCube.read('dataset/VCC1833/big.fits', hdu=2)
# cube = SpectralCube.read('smcube.fits')


for i in tqdm.tqdm(range(0, cube.shape[1], 1)):
    for j in range(0, cube.shape[2], 1):
        cube[:, i, j].quicklook()

plt.show()