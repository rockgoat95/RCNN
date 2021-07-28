import os
import cv2
import matplotlib.pyplot as plt
import numpy as np           
import xml.etree.ElementTree as Et

def image_read(img_path, annot_path, number):
    annot_file = os.listdir(annot_path)[number]
    filename = annot_file.split(".")[0]+".jpg"
    print(filename)
    image = cv2.imread(os.path.join(img_path,filename))
    
    img = image.copy()
    xml =  open(os.path.join(annot_path, annot_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()
    
    # size = root.find('size')
    # width = size.find('width').text
    # height = size.find('height').text
    # channels = size.find('depth').text
    objects = root.findall("object")
    for _object in objects:
        name = _object.find('name').text
        bndbox = _object.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0), 2)
    plt.figure()
    plt.imshow(img)
    plt.show()

    return image


def get_iou(bb1, bb2):
    assert bb1['xmin'] < bb1['xmax']
    assert bb1['ymin'] < bb1['ymax']
    assert bb2['xmin'] < bb2['xmax']
    assert bb2['ymin'] < bb2['ymax']

    x_left = max(bb1['xmin'], bb2['xmin'])
    y_top = max(bb1['ymin'], bb2['ymin'])
    x_right = min(bb1['xmax'], bb2['xmax'])
    y_bottom = min(bb1['ymax'], bb2['ymax'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['xmax'] - bb1['xmin']) * (bb1['ymax'] - bb1['ymin'])
    bb2_area = (bb2['xmax'] - bb2['xmin']) * (bb2['ymax'] - bb2['ymin'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def around_context(image, x, y, w, h, p):
    imout = image.copy()
    image_mean = np.mean(imout, axis =(0,1), dtype= np.int)
    
    y_width, x_width,_ = imout.shape

    padded_image = np.full((y_width+2*p,x_width+2*p, 3), image_mean, dtype=np.uint8)
    padded_image[p:(y_width+p), p:(x_width+p), : ] =imout

    context_img = padded_image[y:(y+h+32), x:(x+w+32), :]

    return context_img

def non_max_suppression(box, pred_score, overlapThresh,class_list):
    n = len(box) 
    if n == 0: 
        return []
    pick = []
    predict = np.array(list(map(lambda x: class_list[np.argmax(x)], pred_score)))
    k = 0
    for cl_name in class_list:
        cl_score_total = pred_score[:,k]
        cl_idx = [i  for i in range(n) if predict[i]==cl_name]

        if cl_name =='background':
            continue
        Rem_idx = cl_idx

        while len(Rem_idx)>0:
            cl_score = cl_score_total[Rem_idx]

            temp_best_idx = Rem_idx[np.argmax(cl_score)]
            pick.append(temp_best_idx)

            Rem_idx = np.setdiff1d(Rem_idx, temp_best_idx)
            for i in Rem_idx:
                if get_iou(box[temp_best_idx],box[i]) >overlapThresh:
                    Rem_idx = np.setdiff1d(Rem_idx, i)
        k +=1
    return pick
