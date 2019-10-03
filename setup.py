from setuptools import setup

setup(name='simifucube',
      version='0.1',
      description='A package to produce IFU datacube from galaxy simulations',
      license="MIT",
      author='Michele Mastropietro',
      author_email='michele.mastropietro@ugent.be',
      install_requires=['pynbody', 'astropy', 'pandas', 'matplotlib', 'scipy', 'numpy', 'specutils', 'spectral-cube'],
      packages=['simifucube'],
      scripts=['simifucube/toy_snap/create_toy_snap.py','simifucube/run.py'],
      # TODO make _render compilation automatic
      # entry_points={
      #     'console_scripts': ['sim_duration=simulation.sim_duration:main',
      #                         ]
      #              }
      )
