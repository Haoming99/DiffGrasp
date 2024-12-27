try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'libs.libkdtree.pykdtree.kdtree',
    sources=[
        'libs/libkdtree/pykdtree/kdtree.pyx',
    ],
    # sources=[
    #     'libs/libkdtree/pykdtree/kdtree.c',
    #     'libs/libkdtree/pykdtree/_kdtree_core.c'
    # ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
    include_dirs=[numpy_include_dir],
)

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'libs.libmcubes.mcubes',
    sources=[
        'libs/libmcubes/mcubes.pyx',
        'libs/libmcubes/pywrapper.cpp',
        'libs/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'libs.libmesh.triangle_hash',
    sources=[
        'libs/libmesh/triangle_hash.pyx'
    ],
    libraries=['m'], # Unix-like specific
    include_dirs=[numpy_include_dir],
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'libs.libmise.mise',
    sources=[
        'libs/libmise/mise.pyx'
    ],
    include_dirs=[numpy_include_dir],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'libs.libsimplify.simplify_mesh',
    sources=[
        'libs/libsimplify/simplify_mesh.pyx'
    ],
    include_dirs=[numpy_include_dir],
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'libs.libvoxelize.voxelize',
    sources=[
        'libs/libvoxelize/voxelize.pyx'
    ],
    libraries=['m'], # Unix-like specific
    include_dirs=[numpy_include_dir],
)


# Gather all extension modules
ext_modules = [
    # pykdtree,
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)
