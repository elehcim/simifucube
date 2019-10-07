from setuptools import setup

setup(name='simifucube',
      version='0.1',
      description='A package to produce IFU datacube from SPH galaxy simulations',
      license="MIT",
      author='Michele Mastropietro',
      author_email='michele.mastropietro@ugent.be',
      install_requires=['pynbody', 'astropy', 'pandas', 'matplotlib',
                        'numpy', 'specutils', 'spectral-cube'],
      packages=['simifucube'],
      scripts=['simifucube/toy_snap/create_toy_snap.py','simifucube/toy2cube'],
      # TODO make _render compilation automatic
      entry_points={
          'console_scripts': ['v_sigma_from_h3h4=simifucube.util.v_sigma_from_h3h4:main',
                              'create_particle_distribution_toysnap=simifucube.toy_snap.create_particle_distribution_toysnap:main',
                              'create_ifucube=simifucube.create_ifucube:main',
                              'pipe=simifucube.pipe:main',
                              ]
                   }
      )
