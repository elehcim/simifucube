from setuptools import setup

setup(name='simifucube',
      version='0.1',
      description='A package to produce IFU datacube from N-body/SPH galaxy simulations',
      license="GPLv3",
      author='Michele Mastropietro',
      author_email='michele.mastropietro@ugent.be',
      install_requires=['pynbody', 'astropy', 'pandas', 'matplotlib',
                        'numpy', 'specutils', 'spectral-cube'],
      packages=['simifucube'],
      scripts=['simifucube/toy_snap/create_two_particles_toy_snap.py','simifucube/toy2cube'],
      # TODO make _render compilation automatic
      entry_points={
          'console_scripts': ['v_sigma_from_h3h4=simifucube.util.v_sigma_from_h3h4:main',
                              'create_particle_distribution_toysnap=simifucube.toy_snap.create_particle_distribution_toysnap:main',
                              'create_ifucube=simifucube.create_ifucube:main',
                              'pipe=simifucube.pipe:main',
                              ]
                   },
      classifiers =   [ "Intended Audience :: Developers",
                        "Intended Audience :: Science/Research",
                        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                        "Programming Language :: Python :: 3",
                        "Topic :: Scientific/Engineering :: Astronomy",
                      ],

      )
