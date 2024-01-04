import time
import traceback
import random

import numpy as np
import cv2
import os

from emosaic.utils.image import divide_image_rectangularly, to_vector
from concurrent.futures import ThreadPoolExecutor


def mosaicify(
        target_image, 
        tile_h, tile_w, 
        tile_index, tile_images, 
        verbose=0,
        use_stabilization=False,
        stabilization_threshold=0.95,
        randomness=0.0,
        opacity=0.0,
        best_k=1,
        trim=True,
        uniform_k=True,
        no_duplicates=True,
    ):
    
    try:
        rect_starts = divide_image_rectangularly(target_image, h_pixels=tile_h, w_pixels=tile_w)
        mosaic = np.zeros(target_image.shape)

        if use_stabilization:
            dist_shape = (target_image.shape[0], target_image.shape[1])
            last_dist = np.zeros(dist_shape).astype(np.int32)
            last_dist[:, :] = 2**31 - 1

        if verbose:
            print("We have %d tiles to assign" % len(rect_starts))

        seen_idx = { -1 }

        def process_tile(j, x, y):
            # get our target region & vectorize it
            target = target_image[x : x + tile_h, y : y + tile_w]
            target_h, target_w, _ = target.shape
            v = to_vector(target, tile_h, tile_w)

            # find nearest codebook image
            try:
                if best_k == 1:
                    dist, I = tile_index.search(v, k=1)
                    idx = I[0][0]
                else:
                    if no_duplicates:
                        # we want to avoid duplicates, so first we try the best and check if it was used before
                        # if it was, then we try the next best, and so on. If all the best are used, then we just pick randomly 
                        # insert pics into seen_idx
                        dist, I = tile_index.search(v, k=best_k + 1)
                        distances = dist[0]
                        for i in range(best_k + 1):
                            idx = I[0][i]
                            if idx not in seen_idx:
                                seen_idx.add(idx)
                                break
                        else:
                            idx = random.choice(I[0])
                            seen_idx.add(idx)
                    elif uniform_k:
                        dist, I = tile_index.search(v, k=best_k)
                        idx = random.choice(I[0])
                    else:
                        dist, I = tile_index.search(v, k=best_k + 1)
                        distances = dist[0]
                        deviation_from_max = np.abs(distances - distances.max())
                        weighting = deviation_from_max / deviation_from_max.sum()
                        idx = np.random.choice(I[0], p=weighting)
                        seen_idx.add(idx)
                closest_tile = tile_images[idx]
            except Exception:
                import ipdb; ipdb.set_trace()

            # write into mosaic
            if random.random() < randomness:
                # pick a random tile!
                num_images = len(tile_images)
                rand_idx = random.choice(range(num_images))
                if rand_idx in seen_idx:
                    remaining_idx = set(range(num_images)) - seen_idx
                    try:
                        rand_idx = random.choice(list(remaining_idx))
                    except IndexError:
                        # just pick one anyway
                        rand_idx = random.choice(range(num_images))

                seen_idx.add(rand_idx)
                mosaic[x : x + tile_h, y : y + tile_w] = tile_images[rand_idx]
            else:
                if use_stabilization:
                    if dist < last_dist[x, y] * stabilization_threshold:
                        mosaic[x : x + tile_h, y : y + tile_w] = closest_tile
                else:
                    mosaic[x : x + tile_h, y : y + tile_w] = closest_tile

            # set new last dist
            if use_stabilization:
                last_dist[x, y] = dist

        starttime = time.time()

        # Create a ThreadPoolExecutor with threads by number of processor cores determined dynamically
        cores = os.cpu_count()
        num_threads = cores
        
        executor = ThreadPoolExecutor(max_workers=num_threads)

        # Use the executor to submit the tasks in parallel
        futures = []
        for (j, (x, y)) in enumerate(rect_starts):
            future = executor.submit(process_tile, j, x, y)
            futures.append(future)

        # Wait for all tasks to complete
        for future in futures:
            future.result()

        elapsed = time.time() - starttime
        if verbose:
            # Print processing time
            print("Processing time: %.4f seconds, on %.2f CPU threads" % (elapsed, num_threads))
            # Print number and percentage of images used
            print("Used %d images, %.2f%% of the codebook" % (len(seen_idx), len(seen_idx) / len(tile_images) * 100))            
            
        # should we adjust opacity? 
        if opacity > 0:
            mosaic = cv2.addWeighted(target_image, opacity, mosaic.astype(np.uint8), 1 - opacity, 0)
        else:
            mosaic = mosaic.astype(np.uint8)

        # should we trim the image to only the tiled area?
        if trim:
            (x1, y1), (x2, y2) = rect_starts[0], rect_starts[-1]
            mosaic = mosaic[x1 : x2, y1 : y2]            

        return mosaic, rect_starts, 0

    except Exception:
        print(traceback.format_exc())
        import ipdb; ipdb.set_trace()
        return None, None, None
