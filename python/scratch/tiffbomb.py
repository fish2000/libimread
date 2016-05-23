
from __future__ import print_function

# from pprint import pprint
import im
import sys

pth = "/Users/fish/Downloads/5604697231_dedfe1d13f_o.jpg"

image = im.Image(pth)
array = im.Array(pth)

im.show(image)
im.show(array)

sys.exit(0)