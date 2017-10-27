from setuptools import setup, find_packages, Extension
import subprocess
import os

libraries = ['boost_python']
extra_objects = []


# Most of this is probably overkill

if 'VIRTUAL_ENV' in os.environ:
    boost_lib = os.path.join(os.environ['VIRTUAL_ENV'], 'lib', 'libboost_python.so')
    if os.path.exists(boost_lib):
        # use local copy of boost
        extra_objects.append(boost_lib)
        libraries.remove('boost_python')


def check_output(*popenargs, **kwargs):
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd, output=output)
    return output


geant4_cflags = check_output(['geant4-config', '--cflags']).split()
geant4_libs = check_output(['geant4-config', '--libs']).split()
# For GEANT4.9.4 built without cmake
try:
    clhep_libs = check_output(['clhep-config', '--libs']).split()
except OSError:
    clhep_libs = []

include_dirs = ['src']

##### figure out location of pyublas headers
try:
    from imp import find_module

    file, pathname, descr = find_module("pyublas")
    from os.path import join

    include_dirs.append(join(pathname, "include"))
except:
    pass  # Don't throw exceptions if prereqs not installed yet

#####

if 'VIRTUAL_ENV' in os.environ:
    include_dirs.append(os.path.join(os.environ['VIRTUAL_ENV'], 'include'))
try:
    import numpy.distutils

    include_dirs += numpy.distutils.misc_util.get_numpy_include_dirs()
except:
    pass  # if numpy doesn't exist yet

setup(
    name='NeutronPhysics',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,

    scripts=[],
    ext_modules=[
        Extension('neutrons.neutron_physics',
                  ['neutron_physics.cc'],
                  include_dirs=include_dirs,
                  extra_compile_args=geant4_cflags,
                  extra_link_args=geant4_libs + clhep_libs,
                  extra_objects=extra_objects,
                  libraries=libraries,
                  ),
    ],

    setup_requires=[],
    install_requires=[],

)
