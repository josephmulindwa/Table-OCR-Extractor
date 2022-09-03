'''Python library to detect table in image'''

import cv2 as cv
import numpy as np
import json
import libutil
import libocr

def get_tabular_image(image, blur_thresh=30, kernel_ratio=0.01):
    '''
    return image that highlights tabular structures and lines
    @params:
    blur_thresh : int, default=30
        the threshold above which a blurred pixel value is considered to be part of a line
    kernel_ratio: float
        this value controls what line thicknesses are to be considered
    '''
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    H, W = imgray.shape
    blur = cv.GaussianBlur(imgray, (3, 3), 0)
    ada_threshed = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY_INV, 7, 7)
    # Thresholding to make the areas of interest such as lines clearer.
    _, threshed = cv.threshold(ada_threshed, 120, 255, cv.THRESH_BINARY_INV)
    # detecting lines
    wsize = int(W * kernel_ratio)
    hkernel = np.ones((1, wsize))
    can = threshed.copy()  #<-- use threshed to get text
    z_mask, o_mask = can == 0, can != 0
    can[z_mask] = 255
    can[o_mask] = 0
    errodeh = cv.erode(can, hkernel, iterations=2)
    dillateh = cv.dilate(errodeh, hkernel, iterations=2)
    
    hsize = int(H*kernel_ratio)
    vkernel = np.ones((hsize, 1))
    errodev = cv.erode(can, vkernel, iterations=1)
    dillatev = cv.dilate(errodev, vkernel, iterations=1)
    
    combined = cv.bitwise_or(dillatev, dillateh)
    blurr = cv.GaussianBlur(combined, (5, 5), 5)
    redilate = cv.dilate(blurr, np.ones((3, 3)), iterations=1)
    _, threshed2 = cv.threshold(redilate, blur_thresh, 255, cv.THRESH_BINARY)
    reerode = cv.erode(threshed2, np.ones((5, 5)), iterations=1)
    return reerode

def get_linked_map(bboxes, proximity, method='center', inclusive=True):
    '''
    return the dictionary of bounding boxes in proximity of each bbox
    @params
    proximity: int
        the distance range under which a bounding box is considered to be close to another
    method: str, default='center'
        the method used to calculate proximity
        see "libutil.in_corner_proximity" for more
    inclusive: boolean, defaukt=True:
        whether to include bounding boxes that are at the border of the proximity or not
    '''
    linked_map = dict()
    for x, bbx in enumerate(bboxes):
        linked_map[x] = libutil.in_corner_proximity(bbx, bboxes, proximity, method=method, inclusive=inclusive)
    return linked_map

def get_row_cells(bbox, bboxes, y_error=4):
    '''
    returns the bounding boxes of the row which the current bbox is part of, itself inclusive
    @params:
    bbox: tuple of length 4
    bboxes: array-like:
        a list of bounding boxes to search from
    y_error: int, default=4
        the error range in which the search is performed
    '''
    x, y, w, h = bbox
    ly, hy = y-y_error, y+h+y_error
    row_bbs = []
    for bb in bboxes:
        x0, y0, w_, h_ = bb
        y1 = y0 + h_
        if y0 >= ly and y1 <= hy: # within yrange of bbox
            row_bbs.append(bb)
    if bbox not in row_bbs:
        row_bbs.append(bbox)
    return row_bbs

def get_tabulars(bboxes, proximity, method='center', inclusive=True):
    '''
    group bounding boxes into table-like structures (tabulars)
    proximity: int
        group bounding boxes that are within this proximity of each other
    method: str, default='center'
        the method used to calculate proximity
        see "libutil.in_corner_proximity" for more
    inclusive: boolean, defaukt=True:
        whether to include bounding boxes that are at the border of the proximity or not
    '''
    linkedmap = get_linked_map(bboxes, proximity, method, inclusive)
    tabulars = []

    while len(linkedmap) > 0:
        key = list(linkedmap.keys())[0]
        stack = [key]  # <-- some key in map
        finished = set()

        while len(stack) > 0:
            bbindx = stack.pop()
            if bbindx not in finished:
                for bb in linkedmap[bbindx]:
                    indx = bboxes.index(bb)
                    if indx == bbindx:
                        continue
                    if indx not in finished and indx not in stack:
                        stack.append(indx)
            finished.add(bbindx)
            del linkedmap[bbindx]
        tabulars.append(sorted([bboxes[i] for i in finished]))
    return tabulars

def remove_enclosing_bboxes(bboxes):
    '''removes bounding boxes that enclose other bboxes if any'''
    while True:
        maxw, maxh, maxi = 0, 0, -1
        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox
            if w > maxw and h > maxh:
                maxw, maxh, maxi = w, h, i

        if maxi != -1:
            enclosing = bboxes[maxi]
            inners = 0
            for bbox in bboxes:
                if enclosing != bbox and libutil.bbox_in_bbox(bbox, enclosing, inclusive=True):
                    inners += 1
            if inners == len(bboxes)-1:
                bboxes.pop(maxi)
            else:
                break
        else:
            break
    return bboxes

def get_non_overlapping(bboxes):
    # same as remove enclosing, however this function keeps one cell and 
    # ignores the overlapping one
    valids = []
    while len(bboxes) > 0:
        bbx = bboxes.pop()
        if len(valids) == 0:
            valids.append(bbx)
        overlap = False
        for bby in valids:
            if is_overlap(bbx, bby):
                overlap = True
                break
        if not overlap:
            valids.append(bbx)
    return valids

def get_row_bb(row):
    '''convenient function to get enclosing bounding box of row'''
    return libutil.get_enclosing_bbox(row)

def get_rows_set(tabular, proximity=4, sort=True):
    '''
    group bounding boxes into rows if cell in proximity of a given row
    proximity: int
        group bounding boxes that are within this proximity of each other
    sort : boolean, default=True
        whether to sort the detected rows by row and column
    '''
    rows_set = []
    for bbox in tabular:
        row = get_row_cells(bbox, tabular, y_error=proximity)
        if row not in rows_set:
            rows_set.append(row)
    if sort: # sorts by y ascending
        bbs = [(i, get_row_bb(row)) for i, row in enumerate(rows_set)]
        bbs.sort(key=lambda item : item[1][1]) # sort by y
        rows_set = [rows_set[i] for i in [i for (i, _) in bbs]]
    return rows_set

class Table:
    '''
    A class that formulates a proper and convenient table structure given a list of rows
    @params:
    rows_set : list
        the list of rows containing cells; cells are bounding boxes
    table_id: int, default=0
        the identifier of this table
    '''
    def __init__(self, rows_set, table_id=0):
        self.table_id = table_id
        self.rows_set = rows_set
        dissolve_span = 4
        self.cells = sorted(list(set([bb for row in rows_set for bb in row])))
        self.cellposition_data = None
        self.celltext_data = None
        self.xmin, self.ymin, self.xmax, self.ymax = 1e10, 1e10, 0, 0

        for row in self.rows_set:
            x, y, w, h = get_row_bb(row)
            self.xmin, self.ymin = min(x, self.xmin), min(y, self.ymin)
            self.xmax, self.ymax = max(x+w, self.xmax), max(y+h, self.ymax)
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        
        raw_rowspans = [[y, y+h] for (x, y, w, h) in self.cells]
        raw_colspans = [[x, x+w] for (x, y, w, h) in self.cells]
        rowbounds = libutil.get_bounds(raw_rowspans, span=dissolve_span)
        colbounds = libutil.get_bounds(raw_colspans, span=dissolve_span)
        # row and col spans are the absolute spans/boundaries
        self.rowspans = list(zip(rowbounds, rowbounds[1:]))
        self.colspans = list(zip(colbounds, colbounds[1:]))
        self.ncols, self.nrows = len(self.colspans), len(self.rowspans)
        
        # getting cell span data
        self.cell_colspans, self.cell_rowspans = [], []
        for i, (x, y, w, h) in enumerate(self.cells):
            rstart, rend = None, None
            for j, (sx, ex) in enumerate(self.colspans):
                if rstart is None and sx-dissolve_span <= x and sx+dissolve_span >= x:
                    rstart = j
                if rend is None and ex-dissolve_span <= (x+w) and (ex+dissolve_span) >= (x+w):
                    rend = j
                if rstart is not None and rend is not None:
                    break
            self.cell_colspans.append([rstart, rend+1])
            
            rstart, rend = None, None
            for j, (sy, ey) in enumerate(self.rowspans):
                if rstart is None and sy-dissolve_span <= y and sy+dissolve_span >= y:
                    rstart = j
                if rend is None and ey-dissolve_span <= (y+h) and (ey+dissolve_span) >= (y+h):
                    rend = j
                if rstart is not None and rend is not None:
                    break
            self.cell_rowspans.append([rstart, rend+1])
        self.get_cellposition_data()
   
    def get_cellposition_data(self):
        '''
        returns cell indices along with their positions
        Example description:
        (i, 0, 0, 2, 6) -> the ith cell is in the first row, first column, rowspan=2, columnspan=6
        (i, 0, 6, 1, 2) -> the ith cell is in the first row, 7th absolute column, rowspan=1, columnspan=2
        '''
        self.cellposition_data = []
        for i, cell in enumerate(self.cells):
            rdata, cdata = self.cell_rowspans[i], self.cell_colspans[i]
            row_id, col_id = rdata[0], cdata[0]
            rspan, cspan = rdata[1]-rdata[0], cdata[1]-cdata[0]
            self.cellposition_data.append((i, row_id, col_id, rspan, cspan))
    
    def get_column(self, col):
        '''return indices of cells that belong to the `col`'''
        return [i for (i, _, col_id, cs, rs) in self.cellposition_data if col_id == col]
    
    def get_row(self, row):
        '''return indices of cells that belong to `row`'''
        return [i for (i, row_id, _, cs, rs) in self.cellposition_data if row_id == row]
    
    def fill_from_data(self, worddata, padding=5, max_words_cell=None, erase_data=True):
        '''
        finds text that belongs to each cell of this table using wordata
        return worddata that is outside the table if erase_data=True
        @params:
        worddata : list of tuples
            a collection of tuples containing text, its confidence and position in an image
            where *this table exists
            see "libocr.get_ocr_worddata"
        padding : int, default=5
            a padding added around each cell. This helps detect text better in some images.
        max_words_cell : None or int, default=None
            the maximum number of words in a cell. If exceeded, the cell will be left empty.
        erase_data : bool, default=True
            whether to erase worddata that has been found inside the table and return new worddata
            If False, the original worddata passed as an argument is returned
        '''
        self.celltext_data = []
        for (i, r, c, rs, cs) in self.cellposition_data:
            x, y, w, h = self.cells[i]
            cell_bbox = (x-padding, y-padding, w+padding*2, h+padding*2)
            removables = []
            for wdata in worddata:
                text, conf, sx, sy, ex, ey = wdata
                bbox = (sx, sy, ex-sx, ey-sy)
                if libutil.bbox_in_bbox(bbox, cell_bbox, inclusive=True):
                    removables.append(wdata)
            if max_words_cell is not None and len(removables) > max_words_cell:
                continue
            if erase_data:
                for wd in removables:
                    worddata.remove(wd)
            self.celltext_data.append(removables)
        return worddata 
    
    def fill_from_image(self, image, border_pad=2, white_pad=10, iteration_factor=0.05, min_size=2, word_padding=1):
        '''
        finds text that belongs to each cell of this table using OCR (pytesseract)
        @params:
        image : ndarray
            the image from which to collect text used to fill cells
        border_pad : int, default=3
            the padding used to remove any borders from each cell
        white_pad : int, default=10
            the whitening value added around each (borderless) cell
        word_padding : int, default=1
            the padding added around every word
        min_size : int, default=2
            the minimum size of each detectable word/letter
        iteration_factor:
            see "libocr.get_text_rois"
        ''' 
        self.celltext_data = []
        for (i, r, c, rs, cs) in self.cellposition_data:
            x, y, w, h = self.cells[i]
            # first erode in the roi; to remove table borders
            roi = image[y+border_pad:y+h-border_pad, x+border_pad:x+w-border_pad].copy()
            roi = cv.copyMakeBorder(roi, border_pad, border_pad, border_pad, border_pad, cv.BORDER_CONSTANT, None, value=(255, 255, 255))
            roi = cv.copyMakeBorder(roi, white_pad, white_pad, white_pad, white_pad, cv.BORDER_CONSTANT, None, value=(255, 255, 255))
            rh, rw = roi.shape[:2]                                                   
            bboxes = libocr.get_text_rois(roi, min_roi_size=min_size, remove_lines=False, as_image=False, 
                                          blur_ksize=3, iteration_factor=iteration_factor) 
            bboxes = [(x, y, w, h) for (x, y, w, h) in bboxes if (w != rw) and (h != rh)]

            wdata = libocr.get_ocr_worddata(roi, bboxes, min_conf=None, padding=0)
            p_wdata = []
            wpad = white_pad - word_padding
            for (text, conf, sx, sy, tw, ty) in wdata:
                x_delta, y_delta = (sx - wpad), (sy - wpad)
                p_wdata.append((text, conf, x+x_delta, y+y_delta, tw+word_padding, ty+word_padding))
            self.celltext_data.append(p_wdata)

    def as_json(self, as_dict=False, include_data=True):
        '''
        formulate the JSON format of *this table
        @params:
        include_data : bool, default=True
            whether to include the coordinates data for each word in a cell
        as_dict: bool, default=False:
            whether to return a dictionary or a JSON string
        '''
        tabular_cell_data = []
        assert self.celltext_data is not None, "Cells not filled, use fill_from_data or fill_from_image to fill cells"
        
        for (span_data, text_data) in zip(self.cellposition_data, self.celltext_data):
            i, row_id, col_id, inrowspan, ncolspan = span_data
            x, y, w, h = self.cells[i]
            cell_text = libocr.get_texts(text_data)
            confs = [tdata[1] for tdata in text_data]
            conf_mean = float(np.mean(confs) if len(confs) > 0 else 0.0)
            # 0-indexed 
            cell_cache = {
                "row": row_id,
                "col": col_id,
                "row_span": inrowspan,
                "col_span": ncolspan,
                "xmin": x,
                "ymin": y,
                "xmax": x+w,
                "ymax": y+h,
                "score": conf_mean,
                "text": cell_text
            }
            if include_data:
                cell_cache['data'] = text_data
            tabular_cell_data.append(cell_cache)

        table_dict = {
            "type":"table",
            "table_id":self.table_id,
            "xmin":int(self.xmin),
            "ymin":int(self.ymin),
            "xmax":int(self.xmax),
            "ymax":int(self.ymax),
            "cells":tabular_cell_data
        }
        if as_dict:
            return table_dict
        return json.dumps(table_dict, indent=4)
    
    def split_along(column_index, inclusive=True, include_in_first=True):
        '''split the table along a column, including/excluding the column-of-split'''
        # this function has not been thoroughly tested
        # mode: inclusive && include_in_first
        first = self.colspans[:column_index+1]
        second = []
        if column_index+1 < len(self.colspans):
            second = self.colspans[column_index+1:]
        # mode: inclusive && !include_in_first
        if not include_in_first:
            first = self.colspans[:column_index]
            second = self.colspans[column_index:]
        # mode: !inclusive
        if not inclusive:
            if include_in_first:
                first.pop()
            else:
                second.pop(0)
        # add these so that they aren't empty
        first.append([self.xmax, self.xmax]) 
        second.append([self.xmax, self.xmax]) 
        first, second = np.array(first), np.array(second)
        fx, fw = np.min(first), np.max(first)-np.min(first)
        sx, sw = np.min(second), np.max(second)-np.min(second)
        bbox0, bbox1 = (fx, self.ymin, fw, self.height), (sx, self.ymin, sw, self.height)
        cells = [cell for row in self.rows_set for cell in row]
        tabular_0, tabular_1 = [], []
        for cell in cells:
            if libutil.bbox_in_bbox(cell, bbox0, inclusive=True):
                tabular_0.append(cell)
            elif libutil.bbox_in_bbox(cell, bbox1, inclusive=True):
                tabular_1.append(cell)
        rows_set_0 = get_rows_set(tabular_0, proximity=4, sort=True)
        rows_set_1 = get_rows_set(tabular_1, proximity=4, sort=True)
        table_0, table_1 = Table(rows_set_0), Table(rows_set_1)
        # if a new table's nrows doesn't match the original's n_rows, then things failed
        # since some cell was cut half way
        #print('Nrows :', table_0.nrows, table_1.nrows)
        if table_0.nrows != self.nrows or table_1.nrows != self.nrows:
            return None
        return table_0, table_1
    
    def split_across(row_index, inclusive=True, include_in_first=True):
        '''
        split the table across a row, including/excluding the row-of-split
        @params
        include_in_first: bool, default=True
            whether to include in first or second table partition after splitting
        '''
        rows_set_0, rows_set_1 = [], []
        for i, row in enumerate(self.rows_set):
            if (row_index < i) or (inclusive and include_in_first and row_index==i):
                rows_set_0.append(row)
            elif (row_index > i) or (inclusive and not include_in_first and row_index==i):
                rows_set_1.append(row)
        return Table(rows_set_0), Table(rows_set_1)
    
    def preview(self, image=None, numbered=True, thickness=2, fontscale=0.8):
        '''
        returns a preview image of this table
        @params
        image: ndarray(dim = 3), default=None
            the image on which to preview this table.
            If None, a new image that this table can fit in will be automatically created
        numbered: bool, default=True
            whether to show the cell indices or not
        thickness : int, default=2
            the thickness of the table borders
        fontscale : float, default=0.8
            controls the size of the cell numbering text
        '''
        imm = np.ones((self.height, self.width), dtype=np.uint8) * 255
        if image is not None:
            imm = image.copy()
        if len(imm.shape) > 2:
            imm = cv.cvtColor(imm, cv.COLOR_BGR2GRAY)
        for i, (x, y, w, h) in enumerate(self.cells):
            x = x - self.xmin if image is None else x
            y = y - self.ymin if image is None else y
            cv.rectangle(imm, (x, y), (x+w, y+h), 0, thickness=thickness)
            if numbered:
                cv.putText(imm, str(i), (x, y+h), cv.FONT_HERSHEY_SIMPLEX, fontscale, 180, 2)
        return imm
    
    def print(self, pad=3):
        '''
        prints and returns the detected table
        @params
        pad : int, default = 3
            the spaces to be added around each word/phrase on either side in each cell
        '''
        celldata = self.as_json(include_data=False, as_dict=True)['cells']
        colwidths = [0 for _ in range(self.ncols)]  # stores max col widths
        metadata = []

        for dct in celldata:
            r, c, rs, cs, txt = dct['row'], dct['col'], dct['row_span'], dct['col_span'], dct['text']
            texts = txt.split('\n')
            for j, phrase in enumerate(texts):
                if len(phrase) > colwidths[c]:
                    colwidths[c] = len(phrase)
                metadata.append((r+j, c, c+cs, phrase))
        metadata.sort()
          
        col_widths = [cw+pad for cw in colwidths]
        prevr, cnt = 0, 0  # for default behavior, remove cnt
        string = ''
        for r, cs, ce, txt in metadata:
            if r != prevr:
                string += '|\n'
                cnt = 0
            cw = sum(col_widths[cs:ce]) + (ce-cs-1)

            expected_loc = sum(col_widths[:cs]) + cs
            if cnt != expected_loc:
                string += ' ' * (expected_loc - cnt)
            addend = '|{}'.format(txt.center(cw))
            cnt += len(addend)
            string += addend
            
            prevr = r
        string += '|'
        print(string)

def get_non_contained_tables(ctables):
    '''return tables that are not in other tables i.e large tables'''
    indices = []
    table_bbs = [(t.xmin, t.ymin, t.width, t.height) for t in ctables]
    table_bbs.sort(key=lambda item : (item[2], item[3])) # start from small tables
    for x, bbx in enumerate(table_bbs):
        in_other = False
        for y, bby in enumerate(table_bbs):
            if x == y:
                continue
            if libutil.bbox_in_bbox(bbx, bby):
                in_other = True
                break
        if not in_other:
            indices.append(x)
        elif x in indices:
            indices.remove(x)
    return [ctables[i] for i in indices]

def get_tables(bboxes, proximity_map=None, rows_map=None):
    '''
    returns tables from bounding boxes
    @params
    proximity_map : dict or None
        see "libtable.get_proximity_map"
    rows_map : dict or None
        see "libtable.get_rows_map"
    '''
    tabulars = []
    if proximity_map is None:
        proximity_map = get_proximity_map(bboxes)
    if rows_map is None:
        rows_map = get_rows_map(bboxes)

    for i in range(len(bboxes)):
        rows_set = get_table_of_cell(i, bboxes, proximity_map, rows_map)
        if len(rows_set) == 0:
            continue
        if rows_set not in tabulars:
            tabulars.append(rows_set)

    #print(len(tabulars), 'tabulars')
    ctables = []
    for tabular in tabulars:
        ctables.append(Table(tabular))

    # remove overlapping
    tables = []
    for x, tablex in enumerate(ctables):
        in_other = False
        for y, tabley in enumerate(tables):
            if x == y:
                continue
            inx = tablex.minx >= tabley.minx and tablex.maxx <= tabley.maxx
            iny = tablex.miny >= tabley.miny and tablex.maxx <= tabley.maxy
            if inx and iny:
                in_other = True
                break
        if not in_other:
            tables.append(tablex)
    return tables

def get_tables_in_image(image, blur_thresh=30, min_cell_size=13, proximity=10, min_table_width_ratio=0.2, 
                        min_rows_table=1, max_row_height_ratio=0.3, min_cells_row=1, min_cells_table=2, kernel_ratio=0.01):
    '''
    returns table objects in an image
    @params:
    image : numpy array
        the raw image used to find tables
    min_cell_size : int, default=13
        the minimum size a cell(bbox) can be
    blur_thresh : int, default=30
        used to control whether small rows/lines are considered as part of the table. 
        Adjusting this parameter will result in more or less rows or columns
    proximity : int, default=10
        the distance between cells for them to be considered as part of the same table
    min_table_width_ratio : float, default=0.2
        the ratio of minimum width in relation to the image width that a table must have/exceed
    max_row_height_ratio : float, default=0.3
        the ratio of maximum height in relation to the image height that a row must not exceed
    '''
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    H, W, C = image.shape

    tabular_image = get_tabular_image(image, blur_thresh, kernel_ratio)
    bboxes = libutil.get_contour_bboxes(tabular_image, min_size=min_cell_size)

    tabulars = get_tabulars(bboxes, proximity)
    tabulars = [tabular for tabular in tabulars if len(tabular) >= min_cells_table]  # added
    tabulars = [remove_enclosing_bboxes(tabular) for tabular in tabulars]
    tabulars = [tabular for tabular in tabulars if len(tabular) >= min_cells_table] # added

    # formulating tables from tabulars
    max_row_height = (max_row_height_ratio * H)
    ctables = []
    for tabular in tabulars:
        rows_set = get_rows_set(tabular)
        # filtering rows by height and min_cells_per_row
        rows_sets, miniset = [], []
        for row in rows_set:
            if len(rows_set) < min_rows_table:
                continue
            _, ymin, __, height = libutil.get_enclosing_bbox(row)
            if height > max_row_height:
                if len(miniset) > 0:
                    rows_sets.append(miniset)
                miniset = []
                continue
            if len(row) < min_cells_row:
                if len(miniset) > 0:
                    rows_sets.append(miniset)
                miniset = []
                continue
            miniset.append(row)
        if len(miniset) > 0:
            rows_sets.append(miniset)
        
        for rows_set in rows_sets:
            if len(rows_set) < min_rows_table:
                continue
            table = Table(rows_set)
            ctables.append(table)

    # remove tables contained in other tables
    min_table_width = int(min_table_width_ratio * W)
    tables = get_non_contained_tables(ctables)
    tables = [table for table in tables if table.width>=min_table_width and table.nrows >= min_rows_table]
    return tables

def get_tabled_mask(image, tables, thickness=2):
    '''return grayscale mask image with tables drawn on it'''
    H, W = image.shape[:2]
    imm = np.ones((H, W), dtype=np.uint8) * 255
    for table in tables:
        for (x, y, w, h) in table.cells:
            cv.rectangle(imm, (x, y), (x+w, y+h), 0, thickness=thickness)
    return imm

def get_detabled_image(image, tables, thickness=3, bgcolor=255):
    '''return image with table cells removed'''
    imc = image.copy()
    H, W = imc.shape[:2]
    for table in tables:
        for (x, y, w, h) in table.cells:
            x0, y0 = max(0, x-thickness), max(0, y-thickness)
            x1, y1 = min(x+w+thickness, W-1), min(y+h+thickness, H-1)
            imc[y0:y1, x0:x1] = bgcolor # whiten region
    return imc
