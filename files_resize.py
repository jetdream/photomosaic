import cv2
import os

verbose = True

def resize_file_aspect_ratio(src_file_path, dst_folder, aspect_ratio, threshold=0.05):
    """
    Resizes a file to achieve a desired aspect ratio of file resolution preserving the aspect ratio of the image
    If the file is already the desired aspect ratio, then it is not resized
    If the aspect ratio differs more than a threshold, then the file is skipped
    The resizing is done only by symmetrical cutting off the edges to keep the image centered.
    Incorrect, corrupted or non-existent files are skipped.
    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    if verbose:
        print("Resizing file %s to aspect ratio %.2f" % (src_file_path, aspect_ratio))
    
    try:
        img = cv2.imread(src_file_path)
        h, w, _ = img.shape
        img_aspect_ratio = h / float(w)
        if abs(img_aspect_ratio - aspect_ratio) <= threshold:
            if verbose:
                print("Resizing file %s from aspect ratio %.2f to %.2f" % (src_file_path, img_aspect_ratio, aspect_ratio))
            if img_aspect_ratio > aspect_ratio:
                # cut off the top and bottom
                new_h = int(w * aspect_ratio)
                h_offset = int((h - new_h) / 2)
                img = img[h_offset:h_offset+new_h, :, :]
            else:
                # cut off the left and right
                new_w = int(h / aspect_ratio)
                w_offset = int((w - new_w) / 2)
                img = img[:, w_offset:w_offset+new_w, :]
            cv2.imwrite(dst_folder + "/" + os.path.basename(src_file_path), img)
        else:
            if verbose:
                print("Skipping file %s with aspect ratio %.2f" % (src_file_path, img_aspect_ratio))
    except:
        if verbose:
            print("Skipping file %s" % (src_file_path))       


def resize_files_aspect_ratio(src_folder, dst_folder, aspect_ratio, threshold=0.05):
    """
    Resizes jpg/jpeg (case insensetive) files in a folder to achieve a desired aspect ratio
    If the file is already the desired aspect ratio, then it is not resized
    If the aspect ratio differs more than a threshold, then the file is skipped
    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for file in os.listdir(src_folder):
        if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
            src_file_path = src_folder + "/" + file
            resize_file_aspect_ratio(src_file_path, dst_folder, aspect_ratio, threshold)
       
    

resize_files_aspect_ratio("E:/Media-projects/flat/", "E:/Media-projects/flat-43-full/", 3/4, 0.4);



  