# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []

ext_modules = [
    CUDAExtension('metaras.cuda.load_textures', [
        'metaras/cuda/load_textures_cuda.cpp',
        'metaras/cuda/load_textures_cuda_kernel.cu',
        ]),
    CUDAExtension('metaras.cuda.create_texture_image', [
        'metaras/cuda/create_texture_image_cuda.cpp',
        'metaras/cuda/create_texture_image_cuda_kernel.cu',
        ]),
    CUDAExtension('metaras.cuda.generalized_renderer', [
        'metaras/cuda/generalized_renderer_cuda.cpp',
        'metaras/cuda/generalized_renderer_cuda_kernel.cu',
        ]),
    CUDAExtension('metaras.cuda.voxelization', [
        'metaras/cuda/voxelization_cuda.cpp',
        'metaras/cuda/voxelization_cuda_kernel.cu',
        ]),
    ]

INSTALL_REQUIREMENTS = ['numpy', 'torch>=1.9.0', 'scikit-image', 'tqdm', 'imageio']

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='metaras',
    version='v0.1.0',
    description='MetaRas',
    author='Chenghao Wu',
    author_email='ucabcw8@ucl.ac.uk	',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Theo-Wu/MetaRas',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='MIT License',
    packages=['metaras', 'metaras.cuda', 'metaras.functional'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
