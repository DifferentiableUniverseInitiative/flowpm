from setuptools import setup

setup(name='flowpm',
      version='0.1',
      description='Particle Mesh Simulation in TensorFlow',
      url='https://github.com/modichirag/flowpm',
      author='Chirag Modi',
      author_email='modichirag@berkeley.edu',
      license='MIT',
      packages=['flowpm'],
      install_requires=['astropy', 'scipy', 'tensorflow'],
      tests_require=['fastpm'],
      extras_require={
        'testing':  ["fastpm>=1.2"],
      })
