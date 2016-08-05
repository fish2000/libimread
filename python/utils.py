
from __future__ import print_function
from distutils.spawn import find_executable as which
from distutils.sysconfig import get_python_inc

# get_terminal_size(): does what you think it does
# adapted from this: http://stackoverflow.com/a/566752/298171
def get_terminal_size(default_LINES=25, default_COLUMNS=80):
    """ Get the width and height of the terminal window in characters """
    import os
    env = os.environ
    def ioctl_GWINSZ(fd):
        try:
            import fcntl, termios, struct
            cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
        except:
            return
        return cr
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        cr = (env.get('LINES',   default_LINES),
              env.get('COLUMNS', default_COLUMNS))
    return int(cr[1]), int(cr[0])

terminal_width, terminal_height = get_terminal_size()

def terminal_print(message, color='red', asterisk='*'):
    """ Print a string to the terminal, centered and bookended with asterisks """
    from clint.textui.colored import red
    from clint.textui import colored
    colorizer = getattr(colored, color.lower(), red)
    message = " %s " % message.strip()
    asterisks = (terminal_width / 2) - (len(message) / 2)
    print(colorizer("""%(aa)s%(message)s%(ab)s""" % dict(
        aa=asterisk[0] * asterisks,
        ab=asterisk[0] * (asterisks + 1 - (len(message) % 2)),
        message=message)))

def collect_generators(build_dir, target_dir):
    import os, shutil
    libs = hdrs = []
    if not os.path.isdir(build_dir):
        raise ValueError("collect_generators(): invalid build_dir")
    if not os.path.isdir(target_dir):
        raise ValueError("collect_generators(): invalid target_dir")
    for dirpath, dirnames, filenames in os.walk(build_dir):
        if os.path.basename(dirpath).lower().startswith("scratch_"):
            for filename in filenames:
                if filename.lower().endswith('.a'):
                    libs.append(os.path.join(dirpath, filename))
                if filename.lower().endswith('.h'):
                    hdrs.append(os.path.join(dirpath, filename))
    for lib in libs:
        destination = os.path.basename(lib)
        shutil.copy2(lib, os.path.join(target_dir, destination))
    for hdr in hdrs:
        destination = os.path.basename(hdr)
        shutil.copy2(lib, os.path.join(target_dir, destination))
    return len(libs)

def list_generator_libraries(target_dir):
    import os
    if not os.path.isdir(target_dir):
        raise ValueError("list_generator_libraries(): invalid target_dir")
    out = []
    for target_file in os.listdir(target_dir):
        if target_file.lower().endswith('.a'):
            out.append(os.path.join(target_dir, target_file))
    return out

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
    import os
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
            # `imread-config --prefix` failed
            raise IOError("command `%s` failed" % cmd)
    
    def bin(self):
        import os
        return os.path.join(self.prefix, "bin")
    def include(self):
        import os
        return os.path.join(self.prefix, "include")
    def lib(self):
        import os
        return os.path.join(self.prefix, "lib")
    def dependency(self, name):
        import os
        return os.path.join(self.prefix, "include", name)

class HomebrewInstall(Install):
    
    def __init__(self, brew_name):
        cmd = "brew --prefix %s" % brew_name
        super(HomebrewInstall, self).__init__(cmd)
