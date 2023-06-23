from setuptools import setup, find_packages

setup(
    name='pointnet',
    packages=find_packages(),
    version='0.0.2',
    license='MIT',
    description='PointNet - Pytorch',
    author='Kaidi Shen',
    url='https://github.com/kentechx/pointnet',
    long_description_content_type='text/markdown',
    keywords=[
        '3D segmentation',
        '3D classification',
        'point cloud understanding',
    ],
    install_requires=[
        'torch>=1.10',
        'einops>=0.6.1',
        'taichi>=1.6.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
