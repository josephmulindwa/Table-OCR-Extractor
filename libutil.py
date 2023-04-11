'''PDF Analysis utility functions and bounding box operations'''
import fitz  # <-- install pymupdf
import numpy as np
import cv2 as cv
import os

PAGE_LIMIT = 10

def image_from_bytes(bytearr):
    img = np.frombuffer(bytearr, dtype=np.uint8)
    return cv.imdecode(img, cv.IMREAD_COLOR)

def page2image(page, zoom=1, return_as='png'):
    '''return scaled pixmap or cv2/numpy image of page'''
    mat = fitz.Matrix(zoom, zoom)
    imf = page.get_pixmap(matrix=mat)
    if return_as is None:
        return imf
    imf_bytes = imf.tobytes(return_as)
    return image_from_bytes(imf_bytes)

def get_zoom_factor(obj, target=1600, by='width'):
    '''
    get a zoom factor such that either width/height >= target
    this function returns the zoom factor such that the output image/pdf will have the `target` size 
    by width/height/any of the two after the pdf as been zoomed
    @params
    by : str, criteria to use in calculating zoom factor
        any -> use the larger of width and height
        height -> use height only
        width -> use width only
    '''
    if type(obj) == str:
        pdfpath = obj
        doc = fitz.open(pdfpath)
        page = doc.load_page(0) # assumes PDF of uniform size
        return get_zoom_factor(page, target)
    else:
        page = obj
        _, __, w, h = page.rect
        w_factor, h_factor = target/w, target/h
        if by == 'width':
            return w_factor
        elif by == 'height':
            return h_factor
        elif by == 'any':
            return w_factor if w >= h else h_factor
    
def pdf2images(pdfpath, zoom=1, save=False, return_as='png', save_path=None, start=0, end=None):
    '''
    Converts each page into image form, returns paths to images or images themselves
    @params
    pdfpath : str
        path to pdf
    zoom : int, float
        the zoom factor to be applied to pdf pages
    save : bool
        whether to save the images and return their paths or return a list of images in memory
    return_as : str; png or jpg
        the format to save images as
    save_path : str
        the directory where to save the images
    start : int
        the page to start from
    end : int, None
        the page to stop at, exclusive
    '''
    unext_path = pdfpath.replace('.pdf', '')
    if ".PDF" in pdfpath:
        unext_path = pdfpath.replace('.PDF', '')
    doc = fitz.open(pdfpath)
    images = []
    if save:
        return_as = None
    
    for i, page in enumerate(doc):
        if i<start:
            continue
        if end is not None and i>=end:
            break
        impath = None
        if save_path is None:
            impath = unext_path + "_"+str(i) + '.png'
        else:
            impath = os.path.join(save_path, str(i) + '.png')
        imf = page2image(page, zoom, return_as=return_as)
        if save:
            imf.save(impath)
            images.append(impath)                        
        else:
            images.append(imf)
    return images

def get_contour_bboxes(image,  min_size=13):
    '''find contours and return their bounding boxes'''
    conts, hkeys = cv.findContours(image.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    is_valid_bb = lambda bb : bb[3] >= min_size and bb[2] >= min_size
    bboxes = [cv.boundingRect(cont) for cont in conts if is_valid_bb(cv.boundingRect(cont))]
    return bboxes

def get_point_cross(point):
    '''
    return cross coordinates surrounding this point
    cross coordinates are the coordinates surrounding a point along the x and y axes
    '''
    x, y = point
    return [(y-1, x), (y, x+1), (y+1, x), (y, x-1)]

def get_corners(bb):
    '''returns the corner points of the bb'''
    x, y, w, h = bb
    return [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]

def point_in_bbox(point, bb, inclusive=True):
    '''
    check if point is within bounding box
    inclusive : bool, determines whether to include edge collisions
    '''
    x, y = point
    x0, y0, w, h = bb
    x1, y1 = x0+w, y0+h
    xr = (x0 < x and x < x1) or (inclusive and (x in [x0, x1]))
    yr = (y0 < y and y < y1) or (inclusive and (y in [y0, y1]))
    return xr and yr

def in_corner_proximity(bb, bboxes, proximity, method='center', inclusive=True):
    '''
    find the bounding boxes that are in the proximity of `bb` around the corners
    @params:
    bb: array_like
        the bounding box to use in the search
    bboxes: list
        a list of bounding boxes to search through
    proximity : int
        how far objects have to be to be considered close
    inclusive : bool, default=True
        whether boxes at the proximity edge are considered or not
    method: str, default='center'
        center -> search while using the corners as the central radial search points
        outward -> search outward from the corners
    '''
    x, y, w, h = bb
    this_corners = get_corners(bb)
    corner_sects = None
    if method == 'center':
        corner_sects = []
        for (x, y) in this_corners:
            central_bb = (x-proximity, y-proximity, proximity*2, proximity*2)
            corner_sects.append(central_bb)
    elif method == 'outward':
        sects = [(x-proximity, y-proximity), (x+w, y-proximity), (x+w, y+h), (x-proximity, y+h)]
        corner_sects = [(s[0], s[1], proximity, proximity) for s in sects]
    
    proxims = []
    for bbox in bboxes:
        corners = get_corners(bbox)
        in_proxim = False
        for point in corners:
            for sect in corner_sects:
                if point_in_bbox(point, sect, inclusive=inclusive):
                    in_proxim = True
                    break
        if in_proxim:
            proxims.append(bbox)
    return proxims

def is_overlap(bb0, bb1, inclusive=True):
    '''check if two bounding boxes, bb0 and bb1 overlap'''
    x0, y0, w0, h0 = bb0
    x1, y1, w1, h1 = bb1

    corners1 = get_corners(bb1)
    ovl = False
    for point in corners:
        if point_in_bbox(point, bb0, inclusive=inclusive):
            return True
    return False

def bbox_in_bbox(bb0, bb1, inclusive=True):
    '''checks if bb0 is in bb1'''
    x0, y0, w0, h0 = bb0
    x1, y1, w1, h1 = bb1
    x_match = ((x0>x1) or (inclusive and x1==x0)) and ((x0+w0<x1+w1) or (inclusive and x0+w0==x1+w1))
    y_match = ((y0>y1) or (inclusive and y1==y0)) and ((y0+h0<y1+h1) or (inclusive and y0+h0==y1+h1))
    return x_match and y_match

def in_proximity(bb, bboxes, proximity, inclusive=False):
    '''
    check for bounding boxes in "surrounding" proximity of bb
    This detection does not start at the corners as seen in in_corner_proximity, it starts at every edge
    Overlapping and internal 
    @params
    inclusive : bool, default=True
        include those that are at the proximity edge
    '''
    x, y, w, h = bb
    nbb = x-proximity, y-proximity, w+proximity*2, h+proximity*2
    res = []
    for bbox in bboxes:
        corners = get_corners(bbox)
        for point in corners:
            if point_in_bbox(point, bbox, inclusive=inclusive):
                res.append(bbox)
                break
    return res

def get_enclosing_bbox(bboxes):
    '''get the enclosing rectangle(xmin, ymin, xmax, ymax) of bboxes'''
    xywh = np.array([[x, y, x+w, y+h] for (x, y, w, h) in bboxes])
    xmin, ymin = np.min(xywh[:, 0]), np.min(xywh[:, 1])
    xmax, ymax = np.max(xywh[:, 2]), np.max(xywh[:, 3])
    return xmin, ymin, xmax-xmin, ymax-ymin

def get_bounds(ranges, span=4):
    '''
    returns boundaries/separators given a list of ranges
    A separator is a value that separates (two) values surrounding it
    Use case:
        Provided with a list of ranges, this function tries to find the bins in which these ranges exist,
        returning the values/lines that separate them. Used in detecting rows and columns of tables.
    @params
    ranges : list
        a list of tuples or pair of values
    span : int, default=4
        the range under which close values will be dissolved into one value/separator
    '''
    pipes, out = [], []
    for pair in ranges:
        pipes += pair
    pipes = sorted(list(set(pipes)))
    prev = None  # recent
    for (i, val) in enumerate(pipes[:-1]):
        if prev is not None and (val - prev) <= span:
            prev = (val + prev)//2
            out[-1] = prev
            continue
        prev = val
        out.append(val)
    if (pipes[-1] - out[-1]) > span:
        out.append(pipes[-1])
    return out

def remove_contained_ranges(ranges, return_containt_dict=False):
    '''
    return macro-ranges that have no containt in them
    @params
    ranges : list
        the list, tuples of ranges
        ensures list of ranges has no ranges that are inside other ranges
    return_containt_dict : bool
    '''
    valid = []
    i = 0
    cont_dict = {i:set() for i in  range(len(ranges))}
    for i, (ss, ee) in enumerate(ranges):
        if i not in cont_dict.keys():
            continue
        for j, (s, e) in enumerate(ranges):
            if i == j or j not in cont_dict.keys():
                continue
            if (s >= ss) and (e <= ee):
                cont_dict[i].add(j)
                del cont_dict[j]
    if return_containt_dict:
        return cont_dict
    return [ranges[i] for i in cont_dict.keys()]

def get_area_of_intersection(bb0, bb1):
    '''returns the area of intersection between bb0 and bb1'''
    axmin, aymin, aw, ah = bb0
    axmax, aymax = axmin + aw, aymin + ah
    bxmin, bymin, bw, bh = bb1
    bxmax, bymax = bxmin + bw, bymin + bh
    dx = min(axmax, bxmax) - max(axmin, bxmin)
    dy = min(aymax, bymax) - max(aymin, bymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    
def get_area_of_union(bb0, bb1):
    '''returns union area if the bboxes overlap'''
    _, __, aw, ah = bb0
    _, __, bw, bh = bb1
    aoi = get_area_of_intersection(bb0, bb1)
    if aoi is not None:
        area_a, area_b = (aw * ah), (bw * bh)
        return area_a + (area_b - aoi)

def get_bbox_distance(bbox0, bbox1):
    '''
    returns a tuple (x-distance, y-distance) between bboxes
    '''
    x0, y0, w0, h0 = bbox0
    x1, y1, w1, h1 = bbox1
    dx = np.array([x0-x1, x0-(x1+w1), (x0+w0)-x1, (x0+w0)-(x1+w1)])
    dy = np.array([y0-y1, y0-(y1+h1), (y0+y0)-h1, (y0+h0)-(y1+h1)])
    return np.min(np.abs(dx)), np.min(np.abs(dy))

def add_bbox_padding(bbox, padding):
    '''returns a bbox to which padding has been added'''
    x, y, w, h = bbox
    return x-padding, y-padding, w + padding*2, h + padding*2

def get_pos_nums(num):
    '''lists the numbers that make up a number; num'''
    pos_nums = []
    while num != 0:
        pos_nums.append(num % 10)
        num = num // 10
    return pos_nums[::-1]

def bbox_in_xrange(bbox, xrange):
    '''checks whether a bounding box is in a given horizontal space/range'''
    x, y, w, h = bbox
    sx, ex = xrange
    return (x >= sx and x <= ex) or (x+w >= sx and x+w <= ex)

def bbox_in_hrange(bbox, hrange):
    '''checks whether a bounding box is in a given vertical space/range'''
    x, y, w, h = bbox
    sy, ey = hrange
    return (y >= sy and y <= ey) or (y+h >= sy and y+h <= ey)
