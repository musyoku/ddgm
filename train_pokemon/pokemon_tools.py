# -*- coding: utf-8 -*-
import gzip, os, six, sys
import numpy as np
from six.moves.urllib import request
from PIL import Image
from StringIO import StringIO

url = "http://i.imgur.com/H14ZUQb.png"
image_dir = "images"
filename = "pokemon.png"

try:
	os.mkdir(image_dir)
except:
	pass

print("Downloading {}...".format(url))
request.urlretrieve(url, filename)

f = open(filename, "rb")
all_pokemons = np.asarray(Image.open(StringIO(f.read())).convert("RGBA"), dtype=np.uint8)
f.close()

width = 40
height = 30
rows = 32
total = 53 * rows + 26
for number in xrange(total):
	column = number % rows
	row = number // rows
	image = all_pokemons[row * height:(row + 1) * height, column * width:(column + 1) * width, :]
	image = Image.fromarray(image)
	image.save("{}/{}.png".format(image_dir, number + 1))
	sys.stdout.write("\rsaving...({:d} / {:d})".format(number, total))
	sys.stdout.flush()
