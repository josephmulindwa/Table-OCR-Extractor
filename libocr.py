'''Library for finding text in image using OCR'''

import numpy as np
import cv2 as cv
import libutil

import pytesseract
from pytesseract import Output

tesseract_path = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
#tesseract_path = "/app/.apt/usr/bin/tesseract" # path to tesseract executable

pytesseract.pytesseract.tesseract_cmd = tesseract_path 
    
def get_text_rois(image, min_roi_size=5, remove_lines=False, kwidth_ratio=0.01, iteration_factor=0.0033, blur_ksize=3, ada_ksize=7, as_image=False):
    '''
    returns the bboxes for the Regions Of Interest (roi) where text has been detected
    This function highlights the locations where text exists in an image
    @params
    image : numpy array, dim=3 or 2
        the raw image with text
    min_roi_size : int, default=5
        the minimum size of a roi bbox
    kwidth_ratio : float, default=0.01
        the horizontal kernel width ratio with respect to image width
        This controls the size of characters that can be detected
    remove_lines: boolean, default=False
        whether to erode/remove lines and table-boundaries before detection
    iteration_factor : float, default=0.0033
        used to determine how many iterations of dilation are used to reconstruct rois after line removal by erosion
        This value is a ratio so that it is by default dependant on the size of the image
    as_image : bool, default=False
        whether to return an image highlighting rois OR to return bboxes of those regions
    '''
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    H, W = imgray.shape
    blur = cv.GaussianBlur(imgray, (blur_ksize, blur_ksize), 0)
    ada_threshed = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY_INV, ada_ksize, ada_ksize)
    _, threshed = cv.threshold(ada_threshed, 120, 255, cv.THRESH_BINARY_INV)
    wsize = int(W * kwidth_ratio)
    hkernel = np.ones((1, wsize))
    can = threshed.copy()
    errodeh = cv.erode(can, hkernel, iterations=2)
    dillateh = cv.dilate(errodeh, hkernel, iterations=2)

    itr = round(W * iteration_factor)
    if not remove_lines:
        itr = itr // 2 
    ksize = 3
    rkernel = np.ones((ksize, ksize))
    reerode = dillateh
    if remove_lines:
        reerode = cv.dilate(dillateh, rkernel, iterations=2)
    redilate = cv.erode(reerode, rkernel, iterations=itr)
    
    # add some dilation to pick colons?
    if as_image:
        return redilate
    return libutil.get_contour_bboxes(redilate, min_size=min_roi_size)

def remove_duplicate_worddata(worddata, iou_thresh = 0.25):
    '''
    removes redundant worddata using intersection over union
    works by finding all worddata that intersect with a certain bbox 
    and picking the one with best confidence
    @params
    worddata : list of tuples
        the worddata from ocr-detection processes
    iou_thresh : int, default=0.25
        worddata is considered to be intersecting if its iou is greater than this value
    '''
    out, removables = list(), set()
    for x, tdatax in enumerate(worddata):
        if x in removables:
            continue
        bb0 = tdatax[-4:]
        removables.add(x)
        best_conf, best_data = tdatax[1], tdatax
        for y, tdatay in enumerate(worddata):
            bb1 = tdatay[-4:]
            aoi = libutil.get_area_of_intersection(bb0, bb1)
            aou = libutil.get_area_of_union(bb0, bb1)
            if aoi is not None:
                iou = aoi/aou
                if iou > iou_thresh:
                    removables.add(y)
                    if tdatay[1] > best_conf:
                        best_conf, bes_data = tdatay[1], tdatay
        out.append(best_data)
    return out

def get_ocr_worddata(image, bboxes=None, min_conf=0, padding=3, config=r'--psm 6 --oem 3'):
    '''
    perform OCR on image and return text, confidence and location bounding box
    @params:
    bboxes: array-like, default=None
        the regions of interest where text is to be searched/detected
        When left as None, OCR is performed on the entire image
    min_conf: float, None; default=50
        the confidence below which extracted text is ignored
    padding: int,
        the padding added to left of every bounding box
    config:
        mode for detecting text
        other values : -l eng --psm 6 --oem 3, --psm 3 --oem 3
    '''
    H, W = image.shape[0], image.shape[1]
    if bboxes is None: # read entire image
        bboxes = [(0, 0, W, H)] 
    p_worddata = []
    for bb in bboxes:
        x0, y0, w, h = bb
        if (W == w) and (H == h):  # image-like bb
            continue
        x1, y1 = x0+w, y0+h
        x0_, y0_ = max(0, x0-padding), max(0, y0)
        x1_, y1_ = min(W-1, x1+padding), min(H-1, y1+padding)
        rh = y1_ - y0_
        roi = image[y0_:y1_, x0_:x1_].copy()        
        outdata = pytesseract.image_to_data(roi, output_type=Output.DICT, config=config)

        for i, conf in enumerate(outdata['conf']):
            conf = float(conf)
            text = outdata['text'][i]
            if conf == -1:
                continue
            rgb = (0, 0, 255)
            if conf < 50:
                rgb = (255, 0, 0)
            
            xx0, yy0 = outdata['left'][i], outdata['top'][i]
            w, h = outdata['width'][i], outdata['height'][i]
            h = min(rh-1, h)

            sx, sy, ex, ey = xx0+x0-padding, yy0+y0, xx0+x0+w-padding, yy0+y0+h
            if min_conf is None or (conf >= min_conf):
                p_worddata.append((text, conf, sx, sy, ex-sx, ey-sy))
    return p_worddata

def textdata_to_lines(worddata, bbox_index=2):
    '''
    returns a list of worddata grouped by lines
    @params
    bbox_index : int, default=2
        the index where the bbox starts in the worddata tuples
    '''
    indices = []
    lines = []
    while True:
        worddata = [worddata[i] for i in range(len(worddata)) if i not in indices]
        indices, of_line =[], []
        if len(worddata) == 0:
            break
        bboxes = [data[bbox_index:bbox_index+4] for data in worddata]
        (xx, yy, ww, hh) = bboxes[0]

        for i, (x, y, w, h) in enumerate(bboxes):
            dh, dy = abs(hh-h), abs(y - yy)
            rintersect = max(hh, h) - dh - dy
            if rintersect > 0:
                of_line.append(worddata[i])
                indices.append(i)
        of_line.sort(key=lambda tpl:tpl[2])
        lines.append([(yy, xx), of_line])
    return [td[1] for td in sorted(lines, key=lambda tpl : tpl[0])]

def get_texts(worddata, bbox_index=2, text_index=0):
    '''
    returns a string organised by lines from the worddata
    @params
    bbox_index : int, default=2
        the index where the bbox starts in the worddata tuples
    text_index : int, default=0
        the index where the text exists in the worddata tuples
    '''
    texts = []
    lines = textdata_to_lines(worddata, bbox_index=bbox_index)
    for line in lines:
        texts.append(' '.join([tdata[text_index] for tdata in line]))
    return '\n'.join(texts)

def get_json_from_worddata(worddata, as_list=True):
    '''
    returns json or list format of worddata
    '''
    out = []
    for (text, conf, x, y, w, h) in worddata:
        djson = {'text':text,
                 'xmin': x, 
                 'ymin': y,
                 'xmax': x+w,
                 'ymax': y+h,
                 'score': conf
                }
        out.append(djson)
    if as_list:
        return out
    return json.dumps(out, indent=4)
