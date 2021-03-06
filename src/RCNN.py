#%%

### 코드에 서툴러서 여러개의 list를 분산하여 관리하였는데 
### dictionary 를 이용하거나 list를 하나로 통합해서 관리하면 더 좋을 것으로 보입니다.
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np           
import xml.etree.ElementTree as Et

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras import Model, optimizers
from keras.applications.vgg16 import VGG16
import math 

from sklearn.linear_model import Ridge

from utils import *

#%%
dataset_path = "C:/RCNN/VOCdevkit/VOC2007/"

os.chdir(dataset_path)

img_path = "JPEGImages"
annot_path = "Annotations"

#%%


#%% selective search implementation
cv2.setUseOptimized(True);
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

im = image_read(img_path, annot_path, 0)
ss.setBaseImage(im)
ss.switchToSelectiveSearchFast()
rects = ss.process()
imOut = im.copy()
for i, rect in enumerate(rects):
    x, y, w, h = rect
    cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
plt.imshow(imOut)

#%% selective search 결과이미지를 모두 저장하면 메모리가 부족하므로 이미지 파일이름과 라벨 박스정보만 저장.


def VOCdataset_with_ss(VOC_path):
    '''
        input: VOC_path ( VOCdevkit/VOC2007/)
        
        VOC dataset을 selective search한 결과를 출력하는 함수
        output 은 이미지 파일이름, selective search 결과 좌표, label , iou 들의 리스트들.

        이런식으로 결과를 편집한 이유는 이미지를 직접 저장하여 사용하면 메모리를 과도하게 많이 사용하기 때문.
    '''
    
    os.chdir(dataset_path)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    train_imgfile=[]
    train_rect = []
    train_labels=[]
    img_path = 'JPEGImages'
    annot_path = 'Annotations'
    train_iou = []


    for e,i in enumerate(os.listdir(annot_path)):
        filename = i.split(".")[0]
        img_file = i.split(".")[0]+".jpg"
        if e%100 == 0:
            print('Success:', e)
        image = cv2.imread(os.path.join(img_path,img_file))
        xml =  open(os.path.join(annot_path, i), "r")
        tree = Et.parse(xml)
        root = tree.getroot()
        objects = root.findall("object")
        imout = image.copy()
        
        bb_values=[]
        for _object in objects:
            bndbox = _object.find('bndbox')
            obj_name = _object.find('name').text
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)
            bb_values.append({"name": obj_name,"xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax})

            train_imgfile.append(filename)
            train_rect.append([xmin,ymin,xmax- xmin,ymax- ymin])
            train_labels.append(obj_name)
            train_iou.append(1)

        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        for e2,result in enumerate(ssresults):
            if e2 < 2000 :
                x,y,w,h = result
                bb_iou = []
                for bb_val in bb_values:
                    bb_iou.append(get_iou(bb_val,{"xmin":x,"xmax":x+w,"ymin":y,"ymax":y+h}))

                candidate_idx = bb_iou.index(max(bb_iou))  
                if bb_iou[candidate_idx] > 0.50 :
                    train_imgfile.append(filename)
                    train_rect.append([x,y,w,h])
                    train_labels.append(bb_values[candidate_idx]['name'])
                    train_iou.append(bb_iou[candidate_idx])

                else :
                    train_imgfile.append(filename)
                    train_rect.append([x,y,w,h])
                    train_labels.append('background')
                    train_iou.append(bb_iou[candidate_idx])
    
    train_imgfile = np.array(train_imgfile)
    train_rect = np.array(train_rect)
    train_labels = np.array(train_labels)
    train_iou = np.array(train_iou)

    return train_imgfile, train_rect, train_labels, train_iou




#%%
imgfile, rect, labels, iou= VOCdataset_with_ss(dataset_path)

np.save('C:/RCNN/imgfile', imgfile) 
np.save('C:/RCNN/rect', rect) 
np.save('C:/RCNN/labels', labels) 
np.save('C:/RCNN/iou', iou) 



#%%
# obj_class =[]
# for e,i in enumerate(os.listdir( annot_path)):
#     xml =  open(os.path.join(annot_path, i), "r")
#     tree = Et.parse(xml)
#     root = tree.getroot()
#     objects = root.findall("object")
#     obj_class = obj_class + [_object.find('name').text for _object in objects ]
#     obj_class = list(set(obj_class))
#     if len(obj_class) == 20:
#         break


obj_class = ['person', # Person
           'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', # Animal
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', # Vehicle
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor' # Indoor
           ]

obj_class = ['background'] + obj_class

#%%
iou = np.load('C:/RCNN/iou.npy' )
imgfile = np.load('C:/RCNN/imgfile.npy' )
labels = np.load('C:/RCNN/labels.npy' )
rect = np.load('C:/RCNN/rect.npy' )


#num = len(imgfile)
# num_train = round(0.90*num)
# train_idx = np.random.choice(np.arange(num), num_train, replace = False)
# valid_idx = np.setdiff1d(np.arange(num), train_idx)


#%% 주변 맥락을 고려해주는 함수와 selective search 의 결과를 RCNN input 에 맞게 출력하는 batch 함수 


def RCNN_batch(imgfile, rect, labels, pos_neg_number, idx, obj_class):
    '''
        RCNN 을 위한 batch 함수

        imgfile : numpy list of image file name
        rect :  numpy list of results of selective search of imgfile
        labels : numpy list of labels of rect in imgfile
        pos_neg_number: list of numbers of postive and negative samples
        idx : numpy list of index set for batch
        obj_class : list of object class

        output: batch samples and labels 
    '''
    
    reindexed_imgfile = imgfile[idx]
    reindexed_rect = rect[idx]
    reindexed_labels = labels[idx]

    idx = np.random.choice(idx, len(reindexed_imgfile),replace = False )

    reindexed_imgfile = imgfile[idx]
    reindexed_rect = rect[idx]
    reindexed_labels = labels[idx]

    train_images=[]
    train_labels=[]
    pos_lag = 0
    neg_lag = 0

    img_path = 'JPEGImages'

    Label_binarized = LabelBinarizer()
    Label_binarized.fit(obj_class)

    for i,filename in enumerate(reindexed_imgfile):
        if (reindexed_labels[i] != 'background' and pos_lag <pos_neg_number[0]) or (reindexed_labels[i] == 'background' and neg_lag < pos_neg_number[1]):

            image = cv2.imread(os.path.join(img_path,filename+'.jpg'))
            x,y,w,h = reindexed_rect[i]

            cropped_arround_img= around_context(image, x,y,w,h,16)
            cropped_arround_img = np.array(cropped_arround_img, dtype = np.uint8)
            resized = cv2.resize(cropped_arround_img, (224,224), interpolation = cv2.INTER_AREA)
        
            train_images.append(resized)
            train_labels.append(reindexed_labels[i])

            if reindexed_labels[i] == 'background' :
                neg_lag += 1
            elif reindexed_labels[i] != 'background' :
                pos_lag += 1

        if pos_lag >=pos_neg_number[0] and neg_lag>=pos_neg_number[1]:
            pos_lag = 0 ; neg_lag = 0
            sample_train_images = np.array(train_images)
            sample_train_labels = Label_binarized.transform(train_labels)
            train_images = []
            train_labels = []

            yield (sample_train_images, sample_train_labels)
##%


#%% pretrain 된 VGG16을 가져와서 VOC도메인에 맞게 조정한다.
vggmodel = VGG16(weights='imagenet', include_top=True)
vggmodel.summary()

X= vggmodel.layers[-2].output
predictions = Dense(21, activation="softmax")(X)
model_final = Model(inputs = vggmodel.input, outputs = predictions)

opt = optimizers.SGD(lr=0.001)
model_final.compile(loss = tf.keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
model_final.summary()

#%%
from sklearn.preprocessing import LabelBinarizer
batch_train = RCNN_batch(imgfile, rect,labels, (32,96), np.arange(len(imgfile)),obj_class)

hist = model_final.fit(batch_train, epochs= 200, steps_per_epoch= 10)



# batch_train = RCNN_batch(imgfile, rect,labels,train_idx, (32,96), obj_class)
# batch_val = RCNN_batch(imgfile, rect,labels,valid_idx, (8,24), obj_class)
# 
# from keras.callbacks import ModelCheckpoint, EarlyStopping
# checkpoint = ModelCheckpoint("ieeercnn_vgg16_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
# hist = model_final.fit(batch_train, epochs= 200, steps_per_epoch= 10, validation_data = batch_val,  callbacks=[checkpoint,early])

model_final.save('C:/RCNN/my_model.h5')

## 빠른 구현을 위해 validation 생략 validation을 하고싶다면 RCNN_batch 함수에 idx 를 부여해주면됨.

#%%
loaded_model = tf.keras.models.load_model('C:/RCNN/my_model.h5')
# loaded_model = tf.keras.models.load_model('C:/RCNN/ieeercnn_vgg16_1.h5') 
loaded_model.summary()

from keras import Model
X = loaded_model.layers[-2].output

reconstructed_model = Model(inputs = loaded_model.input, outputs = X)
reconstructed_model.summary()



#%%
from sklearn.linear_model import SGDClassifier

hard_neg_sample_idx = np.arange(len(iou))[np.bitwise_or(iou == 1, iou<0.3)]

svm_batch = RCNN_batch(imgfile, rect,labels, (32,96),hard_neg_sample_idx,obj_class)

obj_class_np = np.array(obj_class)
label_operater = LabelBinarizer()
label_operater.fit(obj_class_np)

sgdc = SGDClassifier(loss= 'log')

for x1, x2 in svm_batch:
    X_partial = reconstructed_model.predict(x1)
    inverse_label = label_operater.inverse_transform(x2)
    sgdc.partial_fit(X_partial, inverse_label, classes = obj_class_np)
# 데이터를 한번에 load 하면 메모리가 터짐/ batch sample로 SVM을 적용하기위해 SGDClassifier 사용
# parameter 조정은 생략하겠습니다. 
# default 가 linear SVM 이고 OvR

import pickle
filename = 'C:/RCNN/SVM_for_RCNN.sav'
pickle.dump(sgdc, open(filename, 'wb'))

#%%
sgdc = pickle.load(open('C:/RCNN/SVM_for_RCNN.sav', 'rb'))


#%%
def data_for_bb_fun(imgfile, rect, iou, bb_nn):
    '''
        bounding box regression의 pool5 feature를 만들어주는 함수
        imgfile : list of image file name
        rect : results of selective search
        bb_nn : CNN for pool5 feature

        output:
        lists of pool5 feature, proposal box, and ground truth box
    '''
    
    idx = np.arange(len(iou))[iou>0.7]


    imgfile = imgfile[idx]
    rect= rect[idx]

    pool5_feature=[]
    proposal_box=[]
    gt_box = []

    
    lag = 0
    img_path = 'JPEGImages'
    annot_path = 'Annotations'

    for i in range(len(imgfile)):
        filename = imgfile[i]+".jpg"
        image = cv2.imread(os.path.join(img_path,filename))
        xml =  open(os.path.join(annot_path, imgfile[i]+".xml"), "r")
        tree = Et.parse(xml)
        root = tree.getroot()
        objects = root.findall("object")
        
        bb_values=[]

        x,y,w,h = rect[i]

        temp_iou = []
        for _object in objects:
            bndbox = _object.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)
            bb_values.append({"xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax})
        
        iou_list = np.array([get_iou(bb_val,{"xmin":x,"xmax":x+w,"ymin":y,"ymax":y+h} ) for bb_val in bb_values])
        best_iou_idx = iou_list.argmax()
        temp = bb_values[best_iou_idx]


        gt_box.append([(temp['xmin']+temp['xmax'])/2,(temp['ymin']+temp['ymax'])/2,temp['xmax']-temp['xmin'] , temp['ymax'] - temp['ymin']])

        cropped_arround_img= around_context(image, x,y,w,h,16)
        cropped_arround_img = np.array(cropped_arround_img, dtype = np.uint8)
        resized = cv2.resize(cropped_arround_img, (224,224), interpolation = cv2.INTER_AREA) 
        resized = np.expand_dims(resized, axis = 0)

        pool5_feature.append(bb_nn.predict(resized)[0])
        proposal_box.append([x + w/2,y+h/2,w , h])

    return pool5_feature, proposal_box , gt_box
#%%

bb_nn = Model(inputs = loaded_model.input, outputs = loaded_model.layers[-4].output)
bb_nn.summary()

pool5_feature, proposal_box,gt_box =data_for_bb_fun(annot_path, img_path, bb_nn)
pool5_feature = np.array(pool5_feature)
proposal_box = np.array(proposal_box)
gt_box = np.array(gt_box)

np.save('C:/RCNN/pool5_feature', pool5_feature) 
np.save('C:/RCNN/proposal_box', proposal_box) 
np.save('C:/RCNN/gt_box', gt_box) 


#%%

pool5_feature = np.load('C:/RCNN/pool5_feature.npy')
proposal_box = np.load('C:/RCNN/proposal_box.npy')
gt_box = np.load('C:/RCNN/gt_box.npy')


def bounding_box_regression(proposal, ground_truth ,neural_feature, alpha):
    '''
        바운딩박스 리그레션을 하는 함수
        proposal : result of selective search 
        
        neural_feature: pool5 feature 

        alpha: regularized constant ( 1000 in  this model)


        output: coefficients of bounding box regression
    '''
    
    t_star = np.zeros(ground_truth.shape)

    t_star[:,0]  = (ground_truth[:,0] - proposal[:,0])/proposal[:,2]
    t_star[:,1]  = (ground_truth[:,1] - proposal[:,1])/proposal[:,3]
    t_star[:,2]  = np.log(ground_truth[:,2] /proposal[:,2])
    t_star[:,3]  = np.log(ground_truth[:,3] /proposal[:,3])

    weight = []
    for i in range(4):
        print(i)
        Ridge_lm = Ridge(alpha = alpha)
        Ridge_lm.fit(neural_feature, t_star[:,i])
        weight.append(Ridge_lm.coef_)
    
    weight = np.array(weight)
    return weight

bb_wt = bounding_box_regression(proposal_box, gt_box, pool5_feature, alpha = 1000)


np.save('C:/RCNN/bb_wt', bb_wt) 

#%%

def adjusted_box(proposal, weight , neural_feature ):
    '''
        bounding box regression 결과를 통해 box를 재조정시켜주는 함수
        proposal : result of selective search 
        weight : coefficient of bounding box regression
        neural_feature: pool5 feature 

        output: modified bounding box using bb reg
    '''
    d_star = weight.dot(neural_feature.T)

    pred_x = proposal[2]*d_star[0] + proposal[0]
    pred_y = proposal[3]*d_star[1] + proposal[1]
    pred_w = proposal[2]*np.exp(d_star[2])
    pred_h = proposal[3]*np.exp(d_star[3])

    return [pred_x,pred_y,pred_w,pred_h]

bb_wt = np.load('C:/RCNN/bb_wt.npy')
#%%
def test_time_dectection(image, nn, bb_nn, svm, weight):
    '''
        결과를 실제 적용하는 함수

        nn : cnn for Object detection 
        bb_nn: cnn for pool5 feature ( subset network of nn)
        svm : SVMs learned using hard negative mining
        weight : coefficient of bounding box regression

        output: image with predicted bounding box and label
    '''
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()
    imout = image.copy()
    box = []
    pred_score =[]
    for e,result in enumerate(ssresults):
        if e < 500:
            x,y,w,h = result
            timage = imout[y:y+h,x:x+w]
            cropped_arround_img= around_context(imout, x,y,w,h,16)
            cropped_arround_img = np.array(cropped_arround_img, dtype = np.uint8)
            resized = cv2.resize(cropped_arround_img, (224,224), interpolation = cv2.INTER_AREA)
            img = np.expand_dims(resized, axis=0)

            pred_score.append(svm.predict_proba(nn.predict(img))[0])
            box.append({'xmin':x,'ymin':y,'xmax':x+w,'ymax':y+h})
   
    obj_class = svm.classes_

    predict = np.array(list(map(lambda x: obj_class[np.argmax(x)], pred_score)))
    box = np.array(box)
    pred_score = np.array(pred_score)
    NMS_idx = non_max_suppression(box,pred_score,overlapThresh = 0.1, class_list = obj_class)
    pred_score_max = list(map(lambda x: max(x), pred_score))



    for k in NMS_idx:
        x, y, w, h =ssresults[k]
        timage = imout[y:y+h,x:x+w]
        cropped_arround_img= around_context(imout, x,y,w,h,16)
        cropped_arround_img = np.array(cropped_arround_img, dtype = np.uint8)
        resized = cv2.resize(cropped_arround_img, (224,224), interpolation = cv2.INTER_AREA)
        img = np.expand_dims(resized, axis=0)
        pool5_feature = bb_nn.predict(img)
        x_new, y_new , w_new, h_new = adjusted_box([x+w/2,y+h/2,w,h ], weight, pool5_feature[0])

        image_rec = cv2.rectangle(imout,(int(x_new -w_new/2) ,int(y_new -h_new/2)),(int(x_new +w_new/2) ,int(y_new +h_new/2)),(255,0,0), 2)
        cv2.putText(image_rec, predict[k] + ":" + pred_score_max[k].round(2).astype(str),(int(box[k]['xmin']/2+box[k]['xmax']/2),int(box[k]['ymin']/2 + box[k]['ymax']/2) ), cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)

    plt.imshow(imout)
    plt.show()
#%% Test


image = image_read(img_path, annot_path, 40)

test_time_dectection(image,reconstructed_model,bb_nn,sgdc,bb_wt)