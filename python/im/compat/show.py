#
# The Python Imaging Library.
# $Id$
#
# im.show() drivers
#
# History:
# 2008-04-06 fl   Created
# 2016-04-16 ab   Copypastaed into libimread
#
# Copyright (c) Secret Labs AB 2008.
# Alexander Bohn relinquishes rights to his meddlings.
#
# See the README file for information on usage and redistribution.
#
from __future__ import print_function

from tempfile import NamedTemporaryFile
from copy import copy
import sys, os

if sys.version_info >= (3, 3):
    from shlex import quote
else:
    from pipes import quote

_viewers = []


def register(viewer, order=1):
    # A class registry in Python without metaclasses
    # is as satisfying as an unfrosted sheet cake
    # ... sorry dogg but it's true
    try:
        if issubclass(viewer, Viewer):
            viewer = viewer()
    except TypeError:
        pass  # raised if viewer wasn't a class
    if order > 0:
        _viewers.append(viewer)
    elif order < 0:
        _viewers.insert(0, viewer)


##
# Displays a given image.
#
# @param image An image object.
# @param title Optional title.  Not all viewers can display the title.
# @param **options Additional viewer options.
# @return True if a suitable viewer was found, false otherwise.

def show(image, title=None, **options):
    for viewer in _viewers:
        if viewer.show(image, title=title, **options):
            return True
    return False


##
# Base class for viewers.

class Viewer(object):
    
    # format = None
    format = "jpg"
    prefix = "yo-dogg-"
    
    def pil_show(self, image, **options):
        # save temporary image to disk
        if image.mode[:4] == "I;16":
            # @PIL88 @PIL101
            # "I;16" isn't an 'official' mode, but we still want to
            # provide a simple way to show 16-bit images.
            base = "L"
            # FIXME: auto-contrast if max() > 255?
        else:
            base = Image.getmodebase(image.mode)
        if base != image.mode and image.mode != "1":
            image = image.convert(base)
        
        return self.show_image(image, **options)
    
    def show(self, image, **options):
        """ Save an im.Image or im.Array out to a temporary file, 
            then fire off an external viewer to show it.
        """
        
        s = ".%s" % self.format
        p = self.prefix
        did_show = False
        output = ""
        
        with NamedTemporaryFile(suffix=s, prefix=p, delete=False) as tf:
            # options needs a 'format' entry if we're writing
            # to a Python file object -- so copy and update:
            writeopts = copy(options)
            if not 'format' in writeopts:
                writeopts.update({ 'format' : self.format })
            
            # hey dogg so image.write() could throw a thing,
            # I'm just sayin
            output = image.write(
                file=tf.file,
                options=writeopts)
            
            did_show = self.show_file(
                output is None and tf.name or output,
                **options)
        
        return did_show
    
    def get_format(self, image):
        # return format name, or None to save as PGM/PPM
        return self.format
    
    def get_command(self, filepath, **options):
        raise NotImplementedError
    
    def save_image(self, image):
        # save to temporary file, and return filename
        # DOGG: THIS RIGHT HERE IS SOME PIL-CENTRIC SHIT
        return image._dump(format=self.get_format(image))
    
    def show_image(self, image, **options):
        # display given image
        return self.show_file(self.save_image(image), **options)
    
    def show_file(self, filepath, **options):
        # display given file
        os.system(self.get_command(filepath, **options))
        return True

# --------------------------------------------------------------------

if sys.platform == "win32":
    
    class WindowsViewer(Viewer):
        format = "jpg"
        
        def get_command(self, filepath, **options):
            return ('start "Pillow" /WAIT "%s" '
                    '&& ping -n 2 127.0.0.1 >NUL '
                    '&& del /f "%s"' % (filepath, filepath))
        
    register(WindowsViewer)
    
elif sys.platform == "darwin":
    
    class MacViewer(Viewer):
        format = "jpg"
        app_command = "open -a /Applications/Preview.app"
        
        def get_command(self, filepath, **options):
            # on darwin open returns immediately resulting in the temp
            # file removal while app is opening
            command = "(%s %s; sleep 20; rm -f %s)&" % (self.app_command, quote(filepath),
                                                        quote(filepath))
            return command
    
    register(MacViewer)
    
else:
    
    # unixoids
    
    def which(executable):
        path = os.environ.get("PATH")
        if not path:
            return None
        for dirname in path.split(os.pathsep):
            filename = os.path.join(dirname, executable)
            if os.path.isfile(filename):
                # FIXME: make sure it's executable
                return filename
        return None
    
    class UnixViewer(Viewer):
        def show_file(self, filepath, **options):
            command, executable = self.get_command_ex(filepath, **options)
            command = "(%s %s; rm -f %s)&" % (command, quote(filepath),
                                              quote(filepath))
            os.system(command)
            return 1
    
    # implementations
    
    class DisplayViewer(UnixViewer):
        def get_command_ex(self, filepath, **options):
            # this is part of imagemagick, dogg
            command = executable = "display"
            return command, executable
    
    if which("display"):
        register(DisplayViewer)
    
    class XVViewer(UnixViewer):
        def get_command_ex(self, filepath, title=None, **options):
            # note: xv is pretty outdated. most modern systems have
            # imagemagick's display command instead.
            command = executable = "xv"
            if title:
                command += " -name %s" % quote(title)
            return command, executable
    
    if which("xv"):
        register(XVViewer)


if __name__ == "__main__":
    # usage: python ImageShow.py imagefile [title]
    from im import Image
    print(show(Image(sys.argv[1]), *sys.argv[2:]))
