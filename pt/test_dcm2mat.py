# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 09:16:40 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import os
import re
import numpy
from pydicom.tag import Tag

class MosaicType(object):
    """
    Enum for the possible types of mosaic data
    """
    ASCENDING = 1
    DESCENDING = 2


def _get_asconv_headers(mosaic):
    """
    Getter for the asconv headers (asci header info stored in the dicom)
    """
    asconv_headers = re.findall(r'### ASCCONV BEGIN(.*)### ASCCONV END ###',
                                mosaic[Tag(0x0029, 0x1020)].value.decode(encoding='ISO-8859-1'),
                                re.DOTALL)[0]

    return asconv_headers


def _get_mosaic_type(mosaic):
    """
    Check the extra ascconv headers for the mosaic type based on the slice position
    We always assume axial in this case
    the implementation resembles the last lines of documentation in
    https://www.icts.uiowa.edu/confluence/plugins/viewsource/viewpagesrc.action?pageId=54756326
    """

    ascconv_headers = _get_asconv_headers(mosaic)

    try:
        size = int(re.findall(r'sSliceArray\.lSize\s*=\s*(\d+)', ascconv_headers)[0])

        # get the locations of the slices
        slice_location = [None] * size
        for index in range(size):
            axial_result = re.findall(
                r'sSliceArray\.asSlice\[%s\]\.sPosition\.dTra\s*=\s*([-+]?[0-9]*\.?[0-9]*)' % index,
                ascconv_headers)
            if len(axial_result) > 0:
                axial = float(axial_result[0])
            else:
                axial = 0.0
            slice_location[index] = axial

        # should we invert (https://www.icts.uiowa.edu/confluence/plugins/viewsource/viewpagesrc.action?pageId=54756326)
        invert = False
        invert_result = re.findall(r'sSliceArray\.ucImageNumbTra\s*=\s*([-+]?0?x?[0-9]+)', ascconv_headers)
        if len(invert_result) > 0:
            invert_value = int(invert_result[0], 16)
            if invert_value >= 0:
                invert = True

        # return the correct slice types
        if slice_location[0] <= slice_location[1]:
            if not invert:
                return MosaicType.ASCENDING
            else:
                return MosaicType.DESCENDING
        else:
            if not invert:
                return MosaicType.DESCENDING
            else:
                return MosaicType.ASCENDING
    except:
        print("MOSAIC_TYPE_NOT_SUPPORTED")


def _mosaic_to_block(mosaic):
    """
    Convert a mosaic slice to a block of data by reading the headers, splitting the mosaic and appending
    """
    # get the mosaic type
    mosaic_type = _get_mosaic_type(mosaic)

    # get the size of one tile format is 64p*64 or 80*80 or something similar
    matches = numpy.array(mosaic.AcquisitionMatrix)
    matches = matches[matches!=0] #re.findall(r'(\d+)\D+(\d+)\D*', str(mosaic[Tag(0x0051, 0x100b)].value))[0]

    ascconv_headers = _get_asconv_headers(mosaic)
    size = [int(matches[0]),
            int(matches[1]),
            int(re.findall(r'sSliceArray\.lSize\s*=\s*(\d+)', ascconv_headers)[0])]

    # get the number of rows and columns
    number_x = int(mosaic.Rows / size[0])
    number_y = int(mosaic.Columns / size[1])

    # recreate 2d slice
    data_2d = mosaic.pixel_array
    # create 3d block
    data_3d = numpy.zeros((size[2], size[1], size[0]), dtype=data_2d.dtype)
    # fill 3d block by taking the correct portions of the slice
    z_index = 0
    for y_index in range(0, number_y):
        if z_index >= size[2]:
            break
        for x_index in range(0, number_x):
            if mosaic_type == MosaicType.ASCENDING:
                data_3d[z_index, :, :] = data_2d[size[1] * y_index:size[1] * (y_index + 1),
                                         size[0] * x_index:size[0] * (x_index + 1)]
            else:
                data_3d[size[2] - (z_index + 1), :, :] = data_2d[size[1] * y_index:size[1] * (y_index + 1),
                                                         size[0] * x_index:size[0] * (x_index + 1)]
            z_index += 1
            if z_index >= size[2]:
                break
    # reorient the block of data
    data_3d = numpy.transpose(data_3d, (2, 1, 0))
    
    affine = _create_affine_siemens_mosaic(mosaic)

    return data_3d


def _create_affine_siemens_mosaic(dicom_input):
    """
    Function to create the affine matrix for a siemens mosaic dataset
    This will work for siemens dti and 4d if in mosaic format
    """
    # read dicom series with pds
    dicom_header = dicom_input 

    # Create affine matrix (http://nipy.sourceforge.net/nibabel/dicom/dicom_orientation.html#dicom-slice-affine)
    image_orient1 = numpy.array(dicom_header.ImageOrientationPatient)[0:3]
    image_orient2 = numpy.array(dicom_header.ImageOrientationPatient)[3:6]

    normal = numpy.cross(image_orient1, image_orient2)

    delta_r = float(dicom_header.PixelSpacing[0])
    delta_c = float(dicom_header.PixelSpacing[1])

    image_pos = dicom_header.ImagePositionPatient

    delta_s = dicom_header.SpacingBetweenSlices
    return numpy.matrix(
        [[-image_orient1[0] * delta_c, -image_orient2[0] * delta_r, -delta_s * normal[0], -image_pos[0]],
         [-image_orient1[1] * delta_c, -image_orient2[1] * delta_r, -delta_s * normal[1], -image_pos[1]],
         [image_orient1[2] * delta_c, image_orient2[2] * delta_r, delta_s * normal[2], image_pos[2]],
         [0, 0, 0, 1]])