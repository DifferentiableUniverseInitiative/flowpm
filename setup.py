from setuptools import setup
from io import open

# read the contents of the README file
with open('README.md', encoding="utf-8") as f:
    long_description = f.read()

setup(name='flowpm',
      description='Particle Mesh Simulation in TensorFlow',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/modichirag/flowpm',
      author='Chirag Modi',
      author_email='modichirag@berkeley.edu',
      license='MIT',
      packages=['flowpm'],
      install_requires=['astropy', 'scipy'],#, 'tensorflow<2.0'],
      tests_require=['fastpm'],
      extras_require={
        'testing':  ["fastpm"],
        },
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics'
        ],
      keywords='cosmology machine learning')
