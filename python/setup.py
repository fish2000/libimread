
from __future__ import division, print_function

import sys, os
from pprint import pformat
from clint.textui.colored import cyan

# PYTHON & NUMPY INCLUDES
from utils import Install
# from utils import HomebrewInstall
from utils import get_python_inc, terminal_print

# SETUPTOOLS
print('')
try:
    import setuptools
except:
    terminal_print("SETUPTOOLS NOT FOUND", color='red')
    print('''
import: module setuptools not found.

On linux, the package is often called python-setuptools.''')
    terminal_print("SETUPTOOLS NOT FOUND", color='red')
    sys.exit(1)
else:
    terminal_print("import: module %s found" % setuptools.__name__,
                   color='yellow', asterisk='=')

from setuptools import setup, Extension, find_packages

try:
    import numpy
except ImportError:
    class FakeNumpy(object):
        def get_include(self):
            return "."
    numpy = FakeNumpy()
    terminal_print("NUMPY NOT FOUND (using shim)", color='red')
else:
    terminal_print("import: module %s found" % numpy.__name__,
                   color='yellow', asterisk='=')
    

# VERSION & METADATA
__version__ = "<undefined>"
exec(compile(
    open(os.path.join(
        os.path.dirname(__file__),
        '__version__.py')).read(),
        '__version__.py', 'exec'))

print('')
terminal_print("version: %s" % __version__)

long_description = """ Python bindings for libimread, dogg. """

libimread = Install()
# libhalide = HomebrewInstall('halide')
# libllvm = HomebrewInstall('llvm')

# COMPILATION
DEBUG = os.environ.get('DEBUG', '1')
VERBOSE = os.environ.get('VERBOSE', '1')
COLOR_TRACE = os.environ.get('COLOR_TRACE', '1')

undef_macros = []
auxilliary_macros = []
define_macros = []
define_macros.append(
    ('VERSION', __version__))
define_macros.append(
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'))
define_macros.append(
    ('PY_ARRAY_UNIQUE_SYMBOL', 'YO_DOGG'))

if DEBUG:
    if int(DEBUG) > 2:
        define_macros.append(
            ('IM_DEBUG', DEBUG))
        define_macros.append(
            ('_GLIBCXX_DEBUG', DEBUG))
        define_macros.append(
            ('IM_VERBOSE', VERBOSE))
        define_macros.append(
            ('IM_COLOR_TRACE', COLOR_TRACE))
        auxilliary_macros.append(
            ('IM_DEBUG', DEBUG))
        auxilliary_macros.append(
            ('_GLIBCXX_DEBUG', DEBUG))
        auxilliary_macros.append(
            ('IM_VERBOSE', VERBOSE))
        auxilliary_macros.append(
            ('IM_COLOR_TRACE', COLOR_TRACE))

# undef_macros = ['IM_VERBOSE', 'IM_COLOR_TRACE']

# print('')
terminal_print("debug level: %s" % DEBUG)
terminal_print("verbosity: %s" % VERBOSE)
terminal_print("color trace: %s" % COLOR_TRACE)

include_dirs = [
    libimread.include(),
    # libhalide.include(),
    # libimread.dependency('imagecompression'),
    # libimread.dependency('iod'),
    # libimread.dependency('libdocopt'),
    # libimread.dependency('libguid'),
    numpy.get_include(),
    get_python_inc(plat_specific=1),
    os.path.join(os.path.dirname(__file__), 'im', 'include')]

library_dirs = [
    libimread.lib(),
    # libhalide.lib(),
    # libllvm.lib(),
]

# other_flags = ['-Qunused-arguments']
other_flags = []

extensions = {
    'im': [
        "im/src/pymethods/butteraugli.cpp",
        "im/src/pymethods/detect.cpp",
        "im/src/pymethods/structcode_parse.cpp",
        "im/src/pymethods/typecheck.cpp",
        "im/src/pymethods/pymethods.cpp",
        "im/src/buffer.cpp",
        "im/src/bufferview.cpp",
        "im/src/detail.cpp",
        "im/src/gil.cpp",
        "im/src/gil-io.cpp",
        "im/src/hybrid.cpp",
        "im/src/hybridimage.cpp",
        "im/src/options.cpp",
        "im/src/pybuffer.cpp",
        "im/src/structcode.cpp",
        "im/src/typecode.cpp",
        "im/src/module.cpp"
    ],
}

libraries = ['imread']

print('')
terminal_print("SETUPTOOLS BUILD CONFIGGG", asterisk='=')
print('')
print(cyan(" EXTENSION MODULES: %i" % len(extensions)))
print(cyan(pformat(extensions)))
print('')
print(cyan(" DEFINED MACROS: %i" % len(define_macros)))
print(cyan(pformat(define_macros)))
print('')
print(cyan(" INCLUDE DIRECTORIES: %i" % len(include_dirs)))
print(cyan(pformat(include_dirs)))
print('')
print(cyan(" LINKED LIBRARIES: %i" % len(libraries)))
print(cyan(" " + ", ".join(libraries)))
print('')

terminal_print("SETUPTOOLS BUILD NOW COMMENCING", asterisk='=')
print('')

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
            '-O3',
            '-funroll-loops',
            '-mtune=native',
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
