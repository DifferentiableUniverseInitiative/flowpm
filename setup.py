from setuptools import setup

setup(name='flowpm',
      version='0.1',
      description='Particle Mesh Simulation in TensorFlow',
      url='https://github.com/modichirag/flowpm',
      author='Chirag Modi',
      author_email='modichirag@berkeley.edu',
      license='MIT',
      packages=['flowpm'],
      install_requires=['fastpm', 'astropy', 'tensorflow'])
