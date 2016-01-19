
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
