import os
import SimpleITK as sitk
import pydicom
import numpy as np
import cv2
import glob
from tqdm import tqdm
import scipy.ndimage

# functions for preprocessing

def is_dicom_file(filename):
    '''
    :param filename: path of dicom file
    :return:
    '''
    file_stream = open(filename, 'rb')
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b'DICM':
        return True
    return False

def load_patient(src_dir):
    '''
    :param src_dir: path of dicom file
    :return: dicom list
    '''
    files = os.listdir(src_dir)
    slices = []
    for s in files:
        if is_dicom_file(src_dir + '/' + s):
            instance = pydicom.read_file(src_dir + '/' + s)
            slices.append(instance)
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu_by_simpleitk(dicom_dir):
    '''
        read all dicom file in the folder and extract grayvalue within (-4000 ~ 4000)
    :param src_dir: dicom path
    :return: image array
    '''
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    img_array = sitk.GetArrayFromImage(image)
    img_array[img_array == -2000] = 0
    return img_array


def resample(image,old_spacing, new_spacing=[1,1,1]):
        '''
        normalize voxel spacing
    :param img&spacing
    :return: image array
    '''
    #calcluate factor
    old_spacing=np.array(old_spacing)
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    print("Shape after: ", image.shape)
    return image


def get_cube_from_img(img3d, center_x, center_y, center_z, block_size):
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size

    start_y = max(center_y - block_size / 2, 0)
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    res = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return res


def normalize_hu(image):
    '''
    normalize gray value from(-4000 ~ 4000) to (0~1)
    :param image 
    :return: list image normalized
    '''
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def load_patient_images(src_dir, wildcard="*.*", exclude_wildcards=[]):
    '''
    load all png images of a patients
    :param image 
    :return: 3D array
    '''
    src_img_paths = glob.glob(src_dir + wildcard)
    for exclude_wildcard in exclude_wildcards:
        exclude_img_paths = glob.glob(src_dir + exclude_wildcard)
        src_img_paths = [im for im in src_img_paths if im not in exclude_img_paths]
    src_img_paths.sort()
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in src_img_paths]
    images = [im.reshape((1,) + im.shape) for im in images]
    res = np.vstack(images)
    return res


def save_cube_img(target_path, cube_img, rows, cols):
    '''
        save 3D image as 2D slices
        :param 2D path,3D input
        :return: 2D images
    '''
    assert rows * cols == cube_img.shape[0]
    img_height = cube_img.shape[1]
    img_width = cube_img.shape[1]
    res_img = np.zeros((rows * img_height, cols * img_width), dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height, target_x:target_x + img_width] = cube_img[row * cols + col]

    cv2.imwrite(target_path, res_img)


# preprocess example

if __name__ == '__main__':
    dicom_dir = 'DCM1'
    # read dicom file(dicom tags)
    slices = load_patient(dicom_dir)
    # get spacing of file
    pixel_spacing = slices[0].PixelSpacing
    pixel_spacing.append(slices[0].SliceThickness)
    pixel_spacing[2], pixel_spacing[0] = pixel_spacing[0],pixel_spacing[2]
    print('The dicom spacing : ', pixel_spacing)
    # extract gray values
    image = get_pixels_hu_by_simpleitk(dicom_dir)
    # normalize to 1/1/1
    image = resample(image, pixel_spacing)
    for i in tqdm(range(image.shape[0])):
        img_path = r"C:\Users\God Y\Desktop\3D\DCM1\img"+str(i).rjust(4, '0')+"_i.png"
        # normalize pixel value to [0,1]
        org_img = normalize_hu(image[i])
        # save imgs as gray level imgs
        cv2.imwrite(img_path, org_img * 255)
