
from __future__ import division, print_function

import sys, os
from pprint import pformat
# from clint.textui.colored import red, cyan, white
from clint.textui.colored import red, cyan

# SETUPTOOLS
try:
    import setuptools
except:
    print('''
setuptools not found.

On linux, the package is often called python-setuptools''')
    sys.exit(1)
else:
    print('''
%s found.
    ''' % setuptools.__name__)

# PYTHON & NUMPY INCLUDES
from utils import Install, HomebrewInstall, gosub
from setuptools import setup, Extension, find_packages
from distutils.sysconfig import get_python_inc

try:
    import numpy
except ImportError:
    class FakeNumpy(object):
        def get_include(self):
            return "."
    numpy = FakeNumpy()

# VERSION & METADATA
__version__ = "<undefined>"
exec(compile(
    open(os.path.join(os.path.dirname(__file__), '__version__.py')).read(),
    '__version__.py', 'exec'))
long_description = open('README.md').read()
# local_command = os.path.join('..', 'dist', 'bin', 'imread-config')
# local_command +=  " --prefix"
# print(local_command)
# libimread = Install(local_command)
libimread = Install()
libhalide = HomebrewInstall('halide')
libllvm = HomebrewInstall('llvm')

# COMPILATION
DEBUG = os.environ.get('DEBUG', '1')
USE_PNG = os.environ.get('USE_PNG', '16')
USE_JPEG = os.environ.get('USE_JPEG', '1')
USE_TIFF = os.environ.get('USE_TIFF', '1')
USE_LLVM = os.environ.get('USE_LLVM', '1')
USE_WEBP = os.environ.get('USE_WEBP', '1')
USE_EIGEN = os.environ.get('USE_EIGEN', '0')

undef_macros = []
auxilliary_macros = []
define_macros = []
define_macros.append(
    ('VERSION', __version__))
define_macros.append(
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'))
define_macros.append(
    ('IM_COLOR_TRACE', '1'))
define_macros.append(
    ('IM_VERBOSE', '1'))

if DEBUG:
    # undef_macros = ['NDEBUG', '__OBJC__', '__OBJC2__']
    if int(DEBUG) > 2:
        define_macros.append(
            ('IM_DEBUG', DEBUG))
        define_macros.append(
            ('_GLIBCXX_DEBUG', '1'))
        define_macros.append(
            ('IM_VERBOSE', '1'))
        auxilliary_macros.append(
            ('IM_DEBUG', DEBUG))
        auxilliary_macros.append(
            ('_GLIBCXX_DEBUG', '1'))
        auxilliary_macros.append(
            ('IM_VERBOSE', '1'))

print('')
print('')
print(red(""" %(s)s DEBUGGG LEVEL: %(lv)s %(s)s """ % dict(s='*' * 65, lv=DEBUG)))

include_dirs = [
    libimread.include(),
    libhalide.include(),
    # libimread.dependency('imagecompression'),
    # libimread.dependency('iod'),
    # libimread.dependency('libdocopt'),
    # libimread.dependency('libguid'),
    # libimread.dependency('libsszip'),
    # libimread.dependency('libMABlockClosure'),
    # libimread.dependency('libAFNetworking'),
    numpy.get_include(),
    get_python_inc(plat_specific=1),
    os.path.join(os.path.dirname(__file__), 'im', 'include')]

library_dirs = [
    libimread.lib(),
    libhalide.lib(),
    libllvm.lib()]

other_flags = ['-Qunused-arguments']

# for pth in (
#     '/usr/local/include',
#     '/usr/X11/include'):
#     if os.path.isdir(pth):
#         include_dirs.append(pth)
#
# for pth in (
#     '/usr/lib',
#     '/usr/local/lib',
#     '/usr/X11/lib'):
#     if os.path.isdir(pth):
#         library_dirs.append(pth)

extensions = {
    'im': [
        "im/src/buffer.cpp",
        "im/src/detail.cpp",
        "im/src/gil.cpp",
        "im/src/gil-io.cpp",
        "im/src/halideimage.cpp",
        "im/src/hybrid.cpp",
        "im/src/hybridimage.cpp",
        "im/src/options.cpp",
        "im/src/pybuffer.cpp",
        "im/src/pycapsule.cpp",
        "im/src/structcode.cpp",
        "im/src/typecode.cpp",
        "im/src/module.cpp"
    ],
}

# the basics
# libraries = ['jpeg', 'png', 'z', 'm', 'Halide', 'imread', 'c++']
libraries = ['m', 'Halide', 'imread', 'c++']
# PKG_CONFIG = which('pkg-config')

# the addenda
def parse_config_flags(config, config_flags=None):
    """ Get compiler/linker flags from pkg-config and similar CLI tools """
    if config_flags is None: # need something in there
        config_flags = ['']
    for config_flag in config_flags:
        out, err, ret = gosub(' '.join([config, config_flag]))
        if len(out):
            for flag in out.split():
                if flag.startswith('-std'): # c++ version or library flag -- IGNORE IT!
                    continue
                if flag.startswith('-L'): # link path
                    if os.path.exists(flag[2:]) and flag[2:] not in library_dirs:
                        library_dirs.append(flag[2:])
                    continue
                if flag.startswith('-l'): # library link name
                    if flag[2:] not in libraries:
                        libraries.append(flag[2:])
                    continue
                if flag.startswith('-D'): # preprocessor define
                    macro = flag[2:].split('=')
                    if macro[0] not in dict(define_macros).keys():
                        if len(macro) < 2:
                            macro.append('1')
                        define_macros.append(tuple(macro))
                    continue
                if flag.startswith('-I'):
                    if os.path.exists(flag[2:]) and flag[2:] not in include_dirs:
                        include_dirs.append(flag[2:])
                    continue
                if flag.startswith('-W'): # compiler options -- DONT STRIP THE THINGY:
                    if flag not in other_flags:
                        other_flags.append(flag)
                    continue

print('')

# # if we're using it, ask it how to fucking work it
# if int(USE_EIGEN):
#     print(white(""" imread.ext: Eigen3 support enabled """))
#     parse_config_flags(
#         PKG_CONFIG,
#         ('eigen3 --libs', 'eigen3 --cflags'))
#
# if int(USE_WEBP):
#     print(white(""" imread.IO: WebP support enabled """))
#     parse_config_flags(
#         PKG_CONFIG,
#         ('libwebp --libs', 'libwebp --cflags'))
#
# if int(USE_TIFF):
#     print(white(""" imread.IO: LibTIFF support enabled """))
#     parse_config_flags(
#         PKG_CONFIG,
#         ('libtiff-4 --libs', 'libtiff-4 --cflags'))
#
# if int(USE_JPEG):
#     print(white(""" imread.IO: jpeglib support enabled """))
#
# if int(USE_PNG):
#     print(white(""" imread.IO: libpng16 support enabled """))
#     libpng_pkg = 'libpng'
#     if USE_PNG.strip().endswith('6'):
#         libpng_pkg += '16' # use 1.6
#     elif USE_PNG.strip().endswith('5'):
#         libpng_pkg += '15' # use 1.5
#     parse_config_flags(
#         PKG_CONFIG, (
#             '%s --libs' % libpng_pkg,
#             '%s --cflags' % libpng_pkg))
#
# if int(USE_LLVM):
#     print(white(""" <setup.py>: LibLLVM/Clang++ support enabled """))
#     parse_config_flags(
#         which('llvm-config', "%s%s%s" % (
#             os.environ['PATH'], os.pathsep,
#             '/usr/local/opt/llvm/bin')),
#         ('--ldflags', '--cxxflags',
#          '--libs', '--includedir'))

print('')

print(red(""" %(s)s BUILD CONFIGGG: %(s)s """ % dict(s='*' * 65)))
print('')
print(cyan(" EXTENSION MODULES: %i" % len(extensions)))
print(cyan(pformat(extensions)))
print('')
print(cyan(" DEFINED MACROS: %i" % len(define_macros)))
print(cyan(pformat(define_macros)))
print('')
print(cyan(" LINKED LIBRARIES: %i" % len(libraries)))
print(cyan(" " + ", ".join(libraries)))
print('')

print(red(""" %(s)s BUILD COMMENCING: %(s)s """ % dict(s='*' * 65)))

print('')

# from distutils.extension import Extension
# from distutils.core import setup
# language="c++",
# '-x', 'c++',

ext_modules = []
for key, sources in extensions.iteritems():
    ext_modules.append(Extension("im.%s" % key,
        libraries=map(
            lambda lib: lib.endswith('.dylib') and lib.split('.')[0] or lib,
                libraries),
        library_dirs=library_dirs,
        include_dirs=include_dirs,
        undef_macros=undef_macros,
        define_macros=define_macros,
        sources=sources,
        extra_compile_args=[
            '-O3', '-mtune=native',
            '-std=c++1z',
            '-stdlib=libc++'
        ] + other_flags))

packages = find_packages()
package_dir = {
    'im': 'im',
}
package_data = dict()
classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Multimedia',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'Topic :: Software Development :: Libraries',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: C++',
    'License :: OSI Approved :: MIT License']

setup(name='imread',
    version=__version__,
    description='libim: Imaging Bridge library',
    long_description=long_description,
    author='Alexander Bohn',
    author_email='fish2000@gmail.com',
    license='MIT',
    platforms=['Any'],
    classifiers=classifiers,
    url='http://github.com/fish2000/libimread',
    packages=packages,
    ext_modules=ext_modules,
    package_dir=package_dir,
    package_data=package_data,
    test_suite='nose.collector')
