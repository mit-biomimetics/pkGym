from setuptools import setup, find_packages

print(find_packages())

setup(name='gpuGym',
      packages=setuptools.find_packages(),
      version='0.1.0',
      description='MIT Humanoid',
      author='Biomimetics Lab',
      author_email='sheim@mit.edu',
      python_requires='>=3.7',
      install_requires=['isaacgym',
                      'matplotlib']
      )
