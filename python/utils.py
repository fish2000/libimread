
from __future__ import print_function
from distutils.spawn import find_executable as which
from distutils.sysconfig import get_python_inc
from functools import wraps

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
        ab=asterisk[0] * (asterisks + 0 - (len(message) % 2)),
        message=message)))

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

def back_tick(cmd, ret_err=False, as_str=True, raise_err=None):
    """ Run command `cmd`, return stdout, or stdout, stderr if `ret_err`
    Roughly equivalent to ``check_output`` in Python 2.7
    Parameters
    ----------
    cmd : sequence
        command to execute
    ret_err : bool, optional
        If True, return stderr in addition to stdout.  If False, just return
        stdout
    as_str : bool, optional
        Whether to decode outputs to unicode string on exit.
    raise_err : None or bool, optional
        If True, raise RuntimeError for non-zero return code. If None, set to
        True when `ret_err` is False, False if `ret_err` is True
    Returns
    -------
    out : str or tuple
        If `ret_err` is False, return stripped string containing stdout from
        `cmd`.  If `ret_err` is True, return tuple of (stdout, stderr) where
        ``stdout`` is the stripped stdout, and ``stderr`` is the stripped
        stderr.
    Raises
    ------
    Raises RuntimeError if command returns non-zero exit code and `raise_err`
    is True
    """
    from subprocess import Popen, PIPE
    if raise_err is None:
        raise_err = False if ret_err else True
    cmd_is_seq = isinstance(cmd, (list, tuple))
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=not cmd_is_seq)
    out, err = proc.communicate()
    retcode = proc.returncode
    cmd_str = ' '.join(cmd) if cmd_is_seq else cmd
    if retcode is None:
        proc.terminate()
        raise RuntimeError(cmd_str + ' process did not terminate')
    if raise_err and retcode != 0:
        raise RuntimeError('{0} returned code {1} with error {2}'.format(
                           cmd_str, retcode, err.decode('latin-1')))
    out = out.strip()
    if as_str:
        out = out.decode('latin-1')
    if not ret_err:
        return out
    err = err.strip()
    if as_str:
        err = err.decode('latin-1')
    return out, err

class InstallNameError(Exception):
    """ courtesy of delocate:
        https://github.com/matthew-brett/delocate/blob/master/delocate/tools.py#L276
    """
    pass

def ensure_writable(f):
    """ courtesy of delocate:
        https://github.com/matthew-brett/delocate/blob/master/delocate/tools.py#L90
        decorator to ensure a filename is writable before modifying it
        If changed, original permissions are restored after the decorated modification.
    """
    @wraps(f)
    def modify(filename, *args, **kwargs):
        import os, stat
        m = os.stat(filename).st_mode
        if not m & stat.S_IWUSR:
            os.chmod(filename, m | stat.S_IWUSR)
        try:
            return f(filename, *args, **kwargs)
        finally:
            # restore original permissions
            if not m & stat.S_IWUSR:
                os.chmod(filename, m)
    return modify

@ensure_writable
def add_rpath(filepath, newpath):
    """ courtesy of delocate:
        https://github.com/matthew-brett/delocate/blob/master/delocate/tools.py#L276
        Add rpath `newpath` to library `filename`
    Parameters
    ----------
    filepath : str
        filename of library
    newpath : str
        rpath to add
    """
    gosub(['install_name_tool', '-add_rpath', newpath, filepath])

def _line0_says_object(line0, filename):
    """ courtesy of delocate:
        https://github.com/matthew-brett/delocate/blob/master/delocate/tools.py#L276
    """
    line0 = line0.strip()
    if line0.startswith('Archive :'):
        # nothing to do for static libs
        return False
    if not line0.startswith(filename + ':'):
        raise InstallNameError('Unexpected first line: ' + line0)
    further_report = line0[len(filename) + 1:]
    if further_report == '':
        return True
    if further_report == ' is not an object file':
        return False
    raise InstallNameError(
        'Too ignorant to know what "{0}" means'.format(further_report))

def get_install_id(filename):
    """ courtesy of delocate:
        https://github.com/matthew-brett/delocate/blob/master/delocate/tools.py#L276
    Return install id from library named in `filename`
    Returns None if no install id, or if this is not an object file.
    Parameters
    ----------
    filename : str
        filename of library
    Returns
    -------
    install_id : str
        install id of library `filename`, or None if no install id
    """
    out = back_tick(['otool', '-D', filename])
    lines = out.split('\n')
    if not _line0_says_object(lines[0], filename):
        return None
    if len(lines) == 1:
        return None
    if len(lines) != 2:
        raise InstallNameError('Unexpected otool output ' + out)
    return lines[1].strip()

@ensure_writable
def set_install_id(filename, install_id):
    """ courtesy of delocate:
        https://github.com/matthew-brett/delocate/blob/master/delocate/tools.py#L276
    Set install id for library named in `filename`
    Parameters
    ----------
    filename : str
        filename of library
    install_id : str
        install id for library `filename`
    Raises
    ------
    RuntimeError if `filename` has not install id
    """
    if get_install_id(filename) is None:
        raise InstallNameError('{0} has no install id'.format(filename))
    gosub(['install_name_tool', '-id', install_id, filename])

@ensure_writable
def ranlib(filename):
    import os
    if not os.path.basename(filename).endswith('.a'):
        raise InstallNameError('{0} is not a proper archive'.format(filename))
    gosub(['ranlib', filename])

def collect_generators(build_dir, target_dir):
    import os, shutil
    libs = []
    hdrs = []
    dylibs = []
    
    # sanity-check our directory arguments
    if not os.path.isdir(build_dir):
        raise ValueError("collect_generators(): invalid build_dir")
    if not os.path.isdir(target_dir):
        raise ValueError("collect_generators(): invalid target_dir")
    
    # walk the build directory to find generator output
    for dirpath, dirnames, filenames in os.walk(build_dir):
        if os.path.basename(dirpath).startswith("scratch_"):
            for filename in filenames:
                if filename.endswith('.a'):
                    libs.append(os.path.join(dirpath, filename))
                elif filename.endswith('.h'):
                    hdrs.append(os.path.join(dirpath, filename))
                elif filename.endswith('.dylib'):
                    dylibs.append(os.path.join(dirpath, filename))
    
    # copy files by type to our target directory
    for group in (libs, dylibs, hdrs):
        for item in group:
            destination = os.path.basename(item)
            shutil.copy2(item, os.path.join(target_dir, destination))
    
    # run `ranlib` on the static libraries
    for lib in libs:
        libpth = os.path.join(target_dir, os.path.basename(lib))
        ranlib(libpth)
    
    # add the target directory as a relative rpath to our new dylibs
    for dylib in dylibs:
        dylibpth = os.path.join(target_dir, os.path.basename(dylib))
        install_id = get_install_id(dylibpth)
        add_rpath(dylibpth, target_dir)
        set_install_id(dylibpth, "@rpath/%s" % install_id)
    
    # return the number of generators we successfully collected
    return len(dylibs)

def list_generator_libraries(target_dir):
    import os
    if not os.path.isdir(target_dir):
        raise ValueError("list_generator_libraries(): invalid target_dir")
    out = []
    for target_file in os.listdir(target_dir):
        if target_file.endswith('.dylib'):
            out.append(os.path.join(target_dir, target_file))
    return out

def list_generator_archives(target_dir):
    import os
    if not os.path.isdir(target_dir):
        raise ValueError("list_generator_archives(): invalid target_dir")
    out = []
    for target_file in os.listdir(target_dir):
        if target_file.endswith('.a'):
            out.append(os.path.join(target_dir, target_file))
    return out

library_dirs = libraries = define_macros = include_dirs = other_flags = []

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
