from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_gather',
    ext_modules=[
        CUDAExtension('custom_gather', ['cuda_gather.cu']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
