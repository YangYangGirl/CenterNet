cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t


def box_rmse(a, b):
    return ((a[0] - b[0])**2 +
              (a[1] - b[1])**2 +
              (a[2] - b[2])**2 +
              (a[3] - b[3])**2)

def box_union(a, b):
    i = box_intersection(a, b)
    u = a[2] * a[3] + b[2] * b[3] - i
    return u

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

def box_intersection(a, b):
    w = overlap(a[0], a[2], b[0], b[2])
    h = overlap(a[1], a[3], b[1], b[3])
    if (w < 0 or h < 0):
        return 0
    area = w * h
    return area

def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    right = r1 if r1 < r2 else r2
    return right - left

def convbox_match(
        np.ndarray[DTYPE_t, ndim=2] pred,
        np.ndarray[DTYPE_t, ndim=2] label,
        int num_box,
        int num_class):

    cdef unsigned int batch_size = label.shape[0]
    cdef unsigned int rows = label.shape[1]
    cdef unsigned int cols = label.shape[2]
    cdef unsigned int channels = label.shape[3]

    cdef np.ndarray[DTYPE_t, ndim=1] truth = np.zeros((4), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.zeros((4), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] pred_mask_output = np.zeros((batch_size, rows, cols, channels * num_box), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] label_mask_output = np.zeros((batch_size, rows, cols, channels), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] update_label_output = np.zeros((batch_size, rows, cols, channels), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] row = np.zeros((num_box), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] col = np.zeros((num_box), dtype=DTYPE)

    cdef unsigned int best_index = -1
    cdef unsigned int rowc = -1, colc = -1

    cdef DTYPE_t cur_match = 0, best_match = 0, count = 0
    cdef DTYPE_t avg_dist = 0, avg_cat = 0, avg_allcat = 0, avg_conn = 0, avg_allconn = 0, avg_obj = 0, avg_anyobj = 0
    cdef DTYPE_t pred_conn = 0, pred_conn_m = 0, pred_conn_c = 0, pred_cat_m = 0, pred_cat_c = 0, pred_cat = 0

    # assert(rows == 14);
    # assert(cols == 14);
    for batch in range(batch_size):
        for rowm in range(rows):
            for colm in range(cols):
                for c in range(channels):
                    for i in range(num_box):
                        #每个通道设置num_box个
                        pred_mask_output[batch, rowm, colm, c*num_box+i] = 0
                        label_mask_output[batch, rowm, colm, c] = 0
                        update_label_output[batch, rowm, colm, c] = label[batch, rowm, colm, c]

          
        for rowm in range(rows):
            for colm in range(cols):
                for i in range(2*num_box):
                    avg_anyobj += pred[batch, rowm, colm, i]
            
                if label[batch, rowm, colm, 0] == 1:
                    # match by overlap
                    best_index = -1
                    rowc = -1
                    colc = -1
                    for i in range(rows):
                        if label[batch, rowm, colm, 6+i] == 1:
                            rowc = i
                            break

                    for i in range(cols):
                        if label[batch, rowm, colm, 6+rows+i] == 1:
                            colc = i
                            break


                    xm = (label[batch, rowm, colm, 2] + colm) / cols
                    ym = (label[batch, rowm, colm, 3] + rowm) / rows
                    xc = (label[batch, rowc, colc, 4] + colc) / cols
                    yc = (label[batch, rowc, colc, 5] + rowc) / rows


                    truth[0] = xm
                    truth[1] = ym
                    truth[2] = abs(xm - xc) * 2.0
                    truth[3] = abs(ym - yc) * 2.0
      
                    best_match = -100
                    for i in range(num_box):
                        pred_cat = 0
                        for j in range(num_class):  
                        #for j in range(20):
                            pred_cat_m =  pred[batch, rowm, colm, (6+2*(rows+cols))*num_box+i*num_class+j]
                            pred_cat_c =  pred[batch, rowc, colc, (6+2*(rows+cols)+num_class)*num_box+i*num_class+j]
                            label_cat_m = label[batch, rowm, colm, 6+2*(rows+cols)+j]
                            label_cat_c = label[batch, rowc, colc, 6+2*(rows+cols)+num_class+j]
                            pred_cat += (pred_cat_m - label_cat_m) ** 2
                            pred_cat += (pred_cat_c - label_cat_c) ** 2


                        pred_conn_m = pred[batch, rowm, colm, 6*num_box+i*(rows+cols)+rowc] * pred[batch, rowm, colm, 6*num_box+i*(rows+cols)+rows+colc]
                        pred_conn_c = pred[batch, rowc, colc, (6+rows+cols)*num_box+i*(rows+cols)+rowm] * pred[batch, rowc, colc, (6+rows+cols)*num_box+i*(rows+cols)+rows+colm]
                        pred_conn = (pred_conn_m + pred_conn_c) / 2.0

                        xm = (pred[batch, rowm, colm, ((num_box+i)*2+0) + colm]) / cols
                        ym = (pred[batch, rowm, colm, ((num_box+i)*2+1) + rowm]) / rows
                        xc = (pred[batch, rowc, colc, ((num_box*2+i)*2+0) + colc]) / cols
                        yc = (pred[batch, rowc, colc, ((num_box*2+i)*2+1) + rowc]) / rows

                        out[0] = xm
                        out[1] = ym
                        out[2] = abs(xm - xc) * 2.0
                        out[3] = abs(ym - yc) * 2.0
                        iou = box_iou(out, truth)
                        rmse = box_rmse(out, truth)

                        cur_match = pred_conn * (iou - rmse + 0.1) + 0.1 * (2 - pred_cat)

                        if cur_match > best_match:
                            best_match = cur_match
                            best_index = i


                    assert best_index != -1
                    assert rowc != -1
                    assert colc != -1

                    row[0], row[1] = rowm, rowc
                    col[0], col[1] = colm, colc

                    for n in range(2):
                        pred_mask_output[batch, row[n], col[n], n*num_box+best_index] = 1
                        label_mask_output[batch, row[n], col[n], n] = 1
                        avg_obj += pred[batch, row[n], col[n], n*num_box+best_index]

                        pred_mask_output[batch, row[n], col[n], ((1+n)*num_box+best_index)*2+0] = 1
                        pred_mask_output[batch, row[n], col[n], ((1+n)*num_box+best_index)*2+1] = 1
                        label_mask_output[batch, row[n], col[n], 2*(1+n)+0] = 1
                        label_mask_output[batch, row[n], col[n], 2*(1+n)+1] = 1
                        avg_dist += (pred[batch, row[n], col[n], ((1+n)*num_box+best_index)*2+0] - label[batch, row[n], col[n], 2*(1+n)+0])**2 + (pred[batch, row[n], col[n], ((1+n)*num_box+best_index)*2+1] - label[batch, row[n], col[n], 2*(1+n)+1])**2

                        for i in range(rows+cols):
                            pred_mask_output[batch, row[n], col[n], (6+n*(rows+cols))*num_box+best_index*(rows+cols)+i] = 1
                            label_mask_output[batch, row[n], col[n], 6+n*(rows+cols)+i] = 1
                            if label[batch, row[n], col[n], 6+n*(rows+cols)+i] == 1:
                                avg_conn += pred[batch, row[n], col[n], (6+n*(rows+cols))*num_box+best_index*(rows+cols)+i]
                            avg_allconn += pred[batch, row[n], col[n], (6+n*(rows+cols))*num_box+best_index*(rows+cols)+i]


                        for i in range(num_class):
                            pred_mask_output[batch, row[n], col[n], (6+2*(rows+cols)+n*num_class)*num_box+best_index*num_class+i] = 1
                            label_mask_output[batch, row[n], col[n], 6+2*(rows+cols)+n*num_class+i] = 1
                            if label[batch, row[n], col[n], 6+2*(rows+cols)+n*num_class+i] == 1:
                                avg_cat += pred[batch, row[n], col[n], (6+2*(rows+cols)+n*num_class)*num_box+best_index*num_class+i]
                            avg_allcat += pred[batch, row[n], col[n], (6+2*(rows+cols)+n*num_class)*num_box+best_index*num_class+i]


                        count += 1

      
    if count > 0:
        print("Detection Avg Dist: ", avg_dist/count)
        print(", Pos Cat: ", avg_cat/count)
        print(", All Cat: " ,avg_allcat/(count*num_class))
        print(", Pos Conn: " ,avg_conn/(count*2))
        print(", All Conn: " ,avg_allconn/(count*(rows + cols)))
        print(", Pos Obj: " ,avg_obj/count)
        print(", Any Obj: " ,avg_anyobj/(batch_size*rows*cols*num_box*2))
        print(", count: " ,count)

    return pred_mask_output, label_mask_output, label_mask_output


