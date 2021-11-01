import os
import sys
import openslide
import numpy as np
import xmltodict
from PIL import Image
import collections
import scipy as sp
from skimage import filters
from skimage.morphology import disk, dilation, binary_erosion, remove_small_objects

def imfill_old(test_array, h_max=255):
    input_array = np.copy(test_array) 
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array

def imfill(test_array):
    return sp.ndimage.binary_fill_holes(test_array)

def imbinarize(img, threshold):
    new_img = np.zeros(img.shape)
    new_img[img >= threshold] = 1
    return new_img

def invert(img):
    return np.logical_not(img).astype(int)

def imdilate(x, radius=5):
    """ Return greyscale morphological dilation of an image,
    see `skimage.morphology.dilation <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.dilation>`_.

    Parameters
    -----------
    x : 2D array image.
    radius : int for the radius of mask.
    """
    mask = disk(radius)
    x = dilation(x, selem=mask)
    return x

def imerode(x, radius=3):
    """ Return greyscale morphological erosion of an image,
    see `skimage.morphology.erosion <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.erosion>`_.

    Parameters
    -----------
    x : 2D array image.
    radius : int for the radius of mask.
    """
    mask = disk(radius)
    x = binary_erosion(x, selem=mask)
    return x

def bwareaopen(bw, p, conn):
    return remove_small_objects(bw, min_size=p, connectivity=conn)

def filter_background(rgb_img):
    radius = 5
    gray = rgb_img[:,:,1]
    # imbinarize
    level = filters.threshold_otsu(gray)

    bw = imbinarize(gray, level)
    print("Running imbinarize...")
    # Same with matlab -- dividing line --

    # imfill
    bw = imfill(invert(bw))
    print("Running imfill...")
    # np.savetxt("/Users/choppy/Downloads/python_res_imfill.csv", bw, fmt='%.2f', delimiter="\t")

    # disk --> imerode
    bw = imerode(bw, 3)
    print("Running imerode...")
    # np.savetxt("/Users/choppy/Downloads/python_res_imerode.csv", bw, fmt='%.2f', delimiter="\t")

    # disk --> imdilate
    bw = imdilate(bw, 5)
    print("Running imdilate...")
    # np.savetxt("/Users/choppy/Downloads/python_res_imdilate.csv", bw, fmt='%.2f', delimiter="\t")

    # imfill
    bw = imfill(bw)
    print("Running imfill...")

    # bwareaopen
    bw = bwareaopen(bw, radius * radius, 8)
    print("Running bwareaopen...")

    # np.savetxt("/Users/choppy/Downloads/python_res_bwareaopen.csv", bw, fmt='%.2f', delimiter="\t")

    print('Filter background from tissues, Success...')

    return bw

def split_filepath(filepath):
    path, ext = os.path.splitext(filepath)
    filename = os.path.basename(path)
    dirname = os.path.dirname(path)
    return dirname, filename, ext

def xml2dict(xmlpath):
    with open(xmlpath, 'r') as f:
        obj = xmltodict.parse(f.read())
        return obj

def get_annotation_multicolor_xml(xmlpath):
    color = []
    annotations = xml2dict(xmlpath).get('Annotations').get('Annotation')  # May be an array or an hashmap
    if type(annotations) == collections.OrderedDict:
        annotations = [annotations,]

    init = 0
    annotation_info = []

    for idx in range(0, len(annotations), 1):
        item_annotation = annotations[idx]
        linecolor = int(item_annotation.get('@LineColor'))
        regions = item_annotation.get('Regions').get('Region')  # May be an array or an hashmap

        if type(regions) == collections.OrderedDict:
            regions = [regions,]
        
        for ridx in range(0, len(regions), 1):
            item_region = regions[ridx]
            # Grab ROI ID area and length
            roi_id = float(item_region.get('@Id'))
            roi_area = float(item_region.get('@Area'))
            roi_length = float(item_region.get('@Length'))

            # Find child containing vertices for current ROI
            vertices = item_region.get('Vertices').get('Vertex')

            # Get Vertices for current ROI
            X = []
            Y = []
            for vidx in range(0, len(vertices), 1):
                X.append(float(vertices[vidx].get('@X')))
                Y.append(float(vertices[vidx].get('@Y')))

            pos = init + ridx + 1
            annotation_info.append({
                "roi_id": roi_id,
                "linecolor": linecolor,
                "area": roi_area,
                "length": roi_length,
                "X": X,
                "Y": Y
            })

            init = pos
            print('Num of ROI: %s, linecolor is: %s' % (ridx + 1, linecolor))
        
        color.append(linecolor)
    return color, annotation_info

def find_index(lst, key, value):
    for i in range(0, len(lst)):
        if lst[i].get(key) == value:
            return i
    return -1

def read_region(pointer, x, y, width, height, down_level):
    # print('Read Region: %s, %s, %s, %s, %s' % (x, y, width, height, down_level))
    print('Read Region: ', pointer.level_downsamples, pointer.level_dimensions)
    downsampling_factor = int(pointer.level_downsamples[down_level])
    x = int(x * downsampling_factor)
    y = int(y * downsampling_factor)

    image = pointer.read_region((x, y), down_level, (int(width), int(height)))
    return np.array(image)

def func_wsi_tiling_v3_box(slide_filepath, line_color_value, def_size, save_level, step, dropR, savepath):
    format = '.png'
    scale = 1 / (2 ** save_level)
    pointer = openslide.OpenSlide(slide_filepath)
    _, id, _ = split_filepath(slide_filepath)
    if slide_filepath.endswith('ndpi'):
        down_level = 7  # sampling downingLevel is 7 (2^7=128) in case of Out Of Memory
        fact = 2 ** down_level  # svs is resampling from ndpi due to big size, so svs is 4^ ndpi is 2^
        xmlpath = slide_filepath.replace("ndpi", "xml")
    elif slide_filepath.endswith('svs'):
        down_level = 3  # sampling downingLevel is 5 (2^5=32) in case of Out Of Memory
        fact = 4 ** down_level  # svs is resampling from ndpi due to big size, so svs is 4^ ndpi is 2^
        xmlpath = slide_filepath.replace("svs", "xml")
    else:
        print("Cannot support the file format: %s" % slide_filepath)
        sys.exit(1)

    color, annotation_info = get_annotation_multicolor_xml(xmlpath)
    if line_color_value not in color:
        print(line_color_value, ' is not in XML! Please check the color coder!')
        return

    index = find_index(annotation_info, 'linecolor', line_color_value)
    # Get the first X, Y
    position = np.array([annotation_info[index].get('X'), annotation_info[index].get('Y')])

    for idx in range(0, position.shape[0] - 1, 1):
        print('Now is ROI %s' % (idx + 1))
        p = [position[0], position[1]]

        # Python int will cause truncation instead of rounding.
        pos_start = (np.amin(p,1) / fact + 0.5).astype(int)
        pos_len = (np.amax(p,1) / fact + 0.5).astype(int) - pos_start

        print("Postion: ", p, pos_start, pos_len, fact)

        low_m_roi = read_region(pointer, pos_start[0], pos_start[1], pos_len[0], pos_len[1] + 1, down_level)
        # print(low_m_roi[:,:,0:3])
        tissue_mask = filter_background(low_m_roi[:,:,0:3])
        # tissue_mask = low_m_roi[:,:,0:3]
        print(pos_start, fact, scale, def_size)

        ratio = int(def_size / fact)
        step_ratio = int(step / fact)
        # print(step_ratio, tissue_mask.shape[0] - ratio + 1, tissue_mask.shape[1] - ratio + 1)
        for i in range(0, tissue_mask.shape[0] - ratio + 1, step_ratio):
            for j in range(0, tissue_mask.shape[1] - ratio + 1, step_ratio):
                print("Test:", tissue_mask.shape[0] - ratio + 1, tissue_mask.shape[1] - ratio + 1)
                print("Index:", i + ratio, j + ratio)
                print("Pos: ", pos_start[0] + j + 1, pos_start[1] + i, fact, scale)
                print("Scale: ", def_size * scale, save_level)

                region = tissue_mask[i: i + ratio, j: j + ratio]

                if len(region) != 0 and sum(sum(region)) > (ratio * ratio) * dropR:
                    patch = read_region(pointer, (pos_start[0] + j + 1) * fact * scale, (pos_start[1] + i) * fact * scale,
                                        def_size * scale, def_size * scale, save_level)
                    img = Image.fromarray(patch)
                    # img = img.resize((int(patch.shape[0] * 0.5), int(patch.shape[1] * 0.5)))
                    filename = '%s_%s_%s_%s%s' % (id, i + 1, j + 1, idx + 1, format)
                    img.save('%s/%s' % (savepath, filename))
                    print(filename)
    pointer.close()

def run_slide2patch(xml_filepath, savepath):
    print("Functions loading success...")
    def_hw_size = 512
    save_level = 1  # 0, 1, 2, 3
    step = 512
    keepr = 0.75  # drop_rate; only white region area in a mask patch > w*h*drop_rate can be saved
    line_color_value = 65280

    dirname, filename, ext = split_filepath(xml_filepath)
    casename = filename + '.ndpi'
    slide_filepath = os.path.join(dirname, casename)

    if not os.path.exists(slide_filepath):
        casename = filename + '.svs'
        slide_filepath = os.path.join(dirname, casename)

        if not os.path.exists(slide_filepath):
            print("Cannot find the slide file: %s" % slide_filepath)
            sys.exit(1)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    func_wsi_tiling_v3_box(slide_filepath, line_color_value, def_hw_size, save_level, step, keepr, savepath)

    print("Finish tiling...")

if __name__ == '__main__':
    # run_slide2patch('/Users/choppy/Downloads/FUSCCTNBC/FUSCCTNBC001.xml', '/Users/choppy/Downloads/FUSCCTNBC/FUSCCTNBC001_files/')
    # run_slide2patch("/Users/choppy/Downloads/test_slide/slides/TCGA-A2-A0ST-01Z-00-DX1.xml", '/Users/choppy/Downloads/test_slide/slides/TCGA-A2-A0ST-01Z-00-DX1_files/')
    # run_slide2patch("/Users/choppy/Downloads/test_slide/slides/TEST_SLIDE_001.xml", "/Users/choppy/Downloads/test_slide/slides/TEST_SLIDE_001_files")
    run_slide2patch("/Users/choppy/Downloads/test_slide/slides/FUSCCTNBC488.xml", "/Users/choppy/Downloads/test_slide/slides/TEST_SLIDE_001_files")
