
from __future__ import division, print_function

import sys, os
from pprint import pformat
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
from utils import Install
# from utils import HomebrewInstall
from utils import get_python_inc
from setuptools import setup, Extension, find_packages

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
    open(os.path.join(
        os.path.dirname(__file__),
        '__version__.py')).read(),
        '__version__.py', 'exec'))

long_description = """ Python bindings for libimread, dogg. """

libimread = Install()
# libhalide = HomebrewInstall('halide')
# libllvm = HomebrewInstall('llvm')

# COMPILATION
DEBUG = os.environ.get('DEBUG', '1')

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
            ('_GLIBCXX_DEBUG', '1'))
        define_macros.append(
            ('IM_VERBOSE', '1'))
        define_macros.append(
            ('IM_COLOR_TRACE', '1'))
        auxilliary_macros.append(
            ('IM_DEBUG', DEBUG))
        auxilliary_macros.append(
            ('_GLIBCXX_DEBUG', '1'))
        auxilliary_macros.append(
            ('IM_VERBOSE', '1'))
        auxilliary_macros.append(
            ('IM_COLOR_TRACE', '1'))

# undef_macros = ['IM_VERBOSE', 'IM_COLOR_TRACE']

print('')
print('')
print(red(""" %(s)s DEBUGGG LEVEL: %(lv)s %(s)s """ % dict(s='*' * 65, lv=DEBUG)))

include_dirs = [
    libimread.include(),
    # libhalide.include(),
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
    # libhalide.lib(),
    # libllvm.lib(),
]

other_flags = ['-Qunused-arguments']

extensions = {
    'im': [
        "im/src/buffer.cpp",
        "im/src/detail.cpp",
        "im/src/gil.cpp",
        "im/src/gil-io.cpp",
        "im/src/pymethods/detect.cpp",
        "im/src/pymethods/structcode_parse.cpp",
        "im/src/pymethods/typecheck.cpp",
        "im/src/pymethods/pymethods.cpp",
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

print(red(""" %(s)s BUILD CONFIGGG: %(s)s """ % dict(s='*' * 65)))
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

print(red(""" %(s)s BUILD COMMENCING: %(s)s """ % dict(s='*' * 65)))
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
