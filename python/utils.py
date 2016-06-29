
from __future__ import print_function
from distutils.spawn import find_executable as which
from distutils.sysconfig import get_python_inc
import os

# GOSUB: basicaly `backticks` (cribbed from plotdevice)
def gosub(cmd, on_err=True):
    """ Run a shell command and return the output """
    from subprocess import Popen, PIPE
    shell = isinstance(cmd, basestring)
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=shell)
    out, err = proc.communicate()
    ret = proc.returncode
    if on_err:
        msg = '%s:\n' % on_err if isinstance(on_err, basestring) else ''
        assert ret==0, msg + (err or out)
    return out, err, ret

# library_dirs = libraries = define_macros = include_dirs = other_flags = []

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


class Install(object):
    
    def __init__(self, cmd='imread-config --prefix'):
        self.prefix = '/usr/local'
        out, err, ret = gosub(cmd, on_err=False)
        if ret == 0:
            self.prefix = out.strip()
        if out == '':
            return # `imread-config --prefix` failed
    
    def bin(self):
        return os.path.join(self.prefix, "bin")
    def include(self):
        return os.path.join(self.prefix, "include")
    def lib(self):
        return os.path.join(self.prefix, "lib")
    def dependency(self, name):
        return os.path.join(self.prefix, "include", name)

class HomebrewInstall(Install):
    
    def __init__(self, brew_name):
        super(HomebrewInstall, self).__init__(cmd="brew --prefix %s" % brew_name)
