import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from emosaic.utils.indexing import index_images
from emosaic import mosaicify
from types import SimpleNamespace


"""
Mosaic starter
"""

args = SimpleNamespace()

# Paths
args.target = 'media/example/Shasta-32x48.jpg'  # Image to make mosaic from, resulting mosaic image will have the same resolution as this image
args.codebook_dir = 'E:/Media-projects/flat-full-43/'  # Source folder of images
# args.codebook_dir = 'E:/Media-projects/flat-1k-43/'  # Source folder of images
args.savepath = 'media/output/'  # Where to save image to. Scale/filename is used in formatting.

# Mosaic parameters
args.height_aspect = 3.0  # Height aspect
args.width_aspect = 4.0  # Width aspect
args.scale = 75  # How large to make tiles, resulting size of the tile image is (scale * height_aspect, scale * width_aspect)

#  Strategy for choosing tiles
args.best_k = 50  # Choose tile from top K best matches
args.no_duplicates = True  # If we should avoid duplicates
args.uniform_k = True  # If we should use the same K for all tiles (ignored when no_duplicates=True)
args.randomness = 0.0  # Probability to use random tile (does not honor no_duplicates)

args.no_trim = False  # If we shouldn't trim around the outside
args.detect_faces = False  # If we should only include pictures with faces in them
args.opacity = 0.3  # Opacity of the original photo
args.vectorization_factor = 1  # Downsize the image by this much before vectorizing



# =========================================


print("=== Creating Mosaic Image ===")
print("Images=%s, target=%s, scale=%d, aspect_ratio=%.4f, vectorization=%d, randomness=%.2f, faces=%s" % (
    args.codebook_dir, args.target, args.scale, args.height_aspect / args.width_aspect, 
    args.vectorization_factor, args.randomness, args.detect_faces))

# sizing for mosaic tiles
height, width = int(args.height_aspect * args.scale), int(args.width_aspect * args.scale)
aspect_ratio = height / float(width)

# get target image
target_image = cv2.imread(args.target)

# index all those images
tile_index, _, tile_images = index_images(
    paths='%s/*.jpg' % args.codebook_dir,
    aspect_ratio=aspect_ratio, 
    height=height,
    width=width,
    vectorization_scaling_factor=args.vectorization_factor,
    caching=True,
    use_detect_faces=args.detect_faces,
    nprocesses = os.cpu_count()
)

print("Using %d tile codebook images..." % len(tile_images))

# transform!
mosaic, rect_starts, _ = mosaicify(
    target_image, height, width,
    tile_index, tile_images,
    randomness=args.randomness,
    opacity=args.opacity,
    best_k=args.best_k,
    trim=not args.no_trim,
    verbose=1,
    uniform_k=args.uniform_k,
    no_duplicates=args.no_duplicates,
)

# convert to 8 bit unsigned integers
mosaic_img = mosaic.astype(np.uint8)

# show in notebook, if running inside one
try:
    plt.figure(figsize = (64, 30))
    plt.imshow(mosaic_img[:, :, [2,1,0]], interpolation='nearest')
except:
    pass

# save to disk
filename = os.path.basename(args.target).split('.')[0]
# compose filename to save including all args like '%s-mosaic-scale-%d-k%.2f....jpg'
save_filename = '%s-mosaic.scale-%d.best_k-%d.aspect-%.2f.opacity-%.2f.vectf-%.2f.jpg' % (
    filename, args.scale, args.best_k, args.height_aspect / args.width_aspect, args.opacity, args.vectorization_factor)
savepath = args.savepath + '/' + save_filename
print("Writing mosaic image to '%s' ..." % savepath)
cv2.imwrite(savepath, mosaic_img)

