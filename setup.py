from setuptools import setup

setup(name='tybalt',
      description='train autoencoders with gene expression data',
      long_description='compress gene expression data with unsupervised learning to extract biological patterns',
      author='Gregory Way',
      author_email='gregory.way@gmail.com',
      url='https://github.com/greenelab/tybalt',
      packages=['tybalt'],
      license='BSD 3-Clause License',
      install_requires=['keras', 'tensorflow', 'pandas', 'scikit-learn'],
      python_requires='>=3.4')


