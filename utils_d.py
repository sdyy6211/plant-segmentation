import cv2
from shapely.geometry import Polygon

from PIL import Image
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import models
from torch import nn

import pandas as pd
from PIL import ImageEnhance

import torch
from torchvision import models
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.optim import Adam,lr_scheduler
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset

from sklearn.cluster import DBSCAN

from io import BytesIO
import requests
import time

def save_image(exported_csv,save_dic):
    
    file_length = exported_csv.shape[0]

    for i in range(file_length):
        
        external_id = exported_csv["External ID"][i]
        
        objects = eval(exported_csv['Label'][i])['objects']
        
        for j in range(len(objects)):
            
            
            if 'bbox' not in objects[j].keys():
                
                    
                url = objects[j]['instanceURI']
                print(url)

                title = objects[j]['title']


                while 1:

                    response = requests.get(url).content

                    try:

                        img = Image.open(BytesIO(response))

                        print('.'.join(external_id.split('.')[:-1]),title,'saved successfully')

                    except:

                        print('.'.join(external_id.split('.')[:-1]),title,'retry')

                        continue

                    break

                #time.sleep(1)

                img.save(r'{0}\{1}_{2}.png'.format(save_dic,'.'.join(external_id.split('.')[:-1]),title))
                
            else:
                
                continue
def save_bbox(exported_csv):
    
    names = exported_csv['External ID'].values
    
    bbox_file = pd.DataFrame(columns=['x_name'])
    
    bbox_file['x_name'] = names
    
    file_length = exported_csv.shape[0]
    
    for i in range(file_length):
        
        count = 0
        
        external_id = exported_csv["External ID"][i]
        
        objects = eval(exported_csv['Label'][i])['objects']
        
        for j in range(len(objects)):

            if 'bbox' in objects[j].keys():
                
                title = objects[j]['title']

                rearranged_bbox = [objects[j]['bbox']['left'],objects[j]['bbox']['top'],
                                  objects[j]['bbox']['left']+objects[j]['bbox']['width'],objects[j]['bbox']['top']+objects[j]['bbox']['height']]
                
                dic = {'labels':title,'boxes':rearranged_bbox}
                
                bbox_file.loc[i,'bbox_{0}'.format(count)] = str(dic)
                
                count += 1

                print(external_id.split('.')[0],title,'bbox','saved successfully')
            
            else:
                
                continue
    bbox_file = bbox_file.dropna(axis = 0,how = 'all',subset = bbox_file.columns.difference(['x_name']))
    bbox_file = bbox_file.fillna('NAN')
    return bbox_file

def create_csv(exported_csv,labels_dic):
    
    os.chdir(labels_dic)
    
    names = os.listdir()
    
    data_dic = pd.DataFrame(columns=['x_name'])
    
    data_dic['x_name'] = names
    
    file_length = exported_csv.shape[0]

    for i in range(file_length):

        external_id = exported_csv["External ID"][i]

        objects = eval(exported_csv['Label'][i])['objects']

        for j in range(len(objects)):
            
            if 'bbox' not in objects[j].keys():
                
                title = objects[j]['title']

                file_name = '.'.join(external_id.split('.')[:-1])+'_'+title+'.png'

                data_dic.loc[data_dic['x_name'] == external_id,title] = file_name
            
            else:
                
                continue
    data_dic = data_dic.dropna(axis = 0,how = 'all',subset = data_dic.columns.difference(['x_name']))
    data_dic = data_dic.fillna('NAN')
    return data_dic

def seperate_labels(bbox,input_test_dic,test_dic,cropped_input_test_dic,cropped_download_test_dic):
    cropped_lb = pd.DataFrame()
    index = 0
    for i in range(bbox.shape[0]):
        obs = bbox.iloc[i,1:]

        img_name = bbox.iloc[i,0]

        img = Image.open(r'{0}\{1}'.format(input_test_dic,img_name))

        for j in range(bbox.shape[1]-1):

            if obs[j] != 'NAN':

                label = eval(obs[j])['labels']

                label_name =  img_name.split('.')[0]+'_'+label+'s'

                label_img = Image.open(r'{0}\{1}.png'.format(test_dic,label_name))

                coordinates = eval(obs[j])['boxes']

                cropped_img = img.crop(coordinates)

                cropped_label_img = label_img.crop(coordinates)


                cropped_img.save(r'{0}\{1}_{2}.jpg'.format(cropped_input_test_dic,img_name.split('.')[0],j))
                cropped_lb.loc[index,'x_name'] = '{0}_{1}.jpg'.format(img_name.split('.')[0],j)

                cropped_label_img.save(r'{0}\{1}_{2}.png'.format(cropped_download_test_dic,label_name,j))
                cropped_lb.loc[index,'plants'] = '{0}_{1}.png'.format(label_name,j)
                index+=1
    return cropped_lb

def find_rectangle(image,area_thres,max_count,decay,min_poly, max_poly):
    area_list = []
    rectangle_points_list = []
    polygon_list = []

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(gray,200,255,1)

    contours,h = cv2.findContours(thresh,1,2)
    count = 0
    while 1:
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

            if len(approx)>=min_poly and len(approx)<= max_poly:
                polygon = Polygon(cnt.squeeze())
                #x_max = cnt.squeeze()[:,0].max()
                #x_min = cnt.squeeze()[:,0].min()
                #y_max = cnt.squeeze()[:,1].max()
                #y_min = cnt.squeeze()[:,1].min()
                area = polygon.area/(img.shape[0]*img.shape[1])
                if (area >= area_thres[0]) and (area < area_thres[1]) :
                    #print(area)
                    area_list.append(area)
                    polygon_list.append(polygon)
                    rectangle_points_list.append(cnt)
                    cv2.drawContours(img,[cnt],0,(0,0,255),-1)
        if (len(polygon_list)>0) or (count>= max_count):
            break
        else:
            area_thres[0] -= decay
            area_thres[1] += decay
            #print(area_thres)
            count+=1
            continue
    return img,area_list,polygon_list,rectangle_points_list

def reset_conners(array):
    left_top = [array[:,0].min(),array[:,1].max()]
    right_top = [array[:,0].max(),array[:,1].max()]
    right_bottom = [array[:,0].max(),array[:,1].min()]
    left_bottom = [array[:,0].min(),array[:,1].min()]
    
    conners = [left_top,right_top,right_bottom,left_bottom]
    conners = np.array(conners)
    
    distance = np.zeros((4,4))
    for row_c,i in enumerate(conners[:4]):
        for col_a,j in enumerate(array[:4]):

            distance[row_c,col_a] = np.sum(np.sqrt(np.power(i - j,2)))
    index = np.argmax(distance,axis = 0).reshape(4,1)
    concat = np.concatenate([index,array[:4]],axis = 1)
    sorted_conners = concat[concat[:,0].argsort()][:,1:]
    return sorted_conners

def adjust_position(img_arr,left,upper):
    img_arr[:,0] = img_arr[:,0] + left
    img_arr[:,1] = img_arr[:,1] + upper
    return img_arr

def find_window(img,dlab,transform_deeplab,device,scale = 0.1,max_count = 45,
                area_thres = [0.45,0.55],decay = 0.01,min_poly = 3, max_poly = 50):
    
    validation = transform_deeplab(image_adjust(img)).unsqueeze(0).to(device)
    coordinates,prob = get_region_v2(dlab(validation,interpolate = False).squeeze(),
                                  label = 1,original_size = img.size,scale = scale)
    img,area_list,polygon_list,rectangle_points_list = find_rectangle(image_adjust(img).crop(coordinates[0]),max_count = max_count,
                                                                  area_thres = area_thres,decay = decay,min_poly = min_poly,max_poly = max_poly)
    return img,area_list,polygon_list,coordinates#rectangle_points_list

def outter(polygon,out = False):
    x,y = polygon.minimum_rotated_rectangle.boundary.coords.xy
    array = np.c_[x,y]
    if out == False:
        return reset_conners(array)
    else:
        conners = reset_conners(array[:4])
        array_convex_hull = np.c_[polygon.convex_hull.boundary.coords.xy[0],polygon.convex_hull.boundary.coords.xy[1]]
        distance = np.zeros((4,array_convex_hull.shape[0]))
        for row_c,i in enumerate(conners[:4]):
            for col_a,j in enumerate(array_convex_hull):

                distance[row_c,col_a] = np.sum(np.sqrt(np.power(i - j,2)))
        index = np.argmax(distance,axis = 1)

        return reset_conners(array_convex_hull[index].reshape(4,2))

class GET_LABEL():
    
    def __init__(self,size):
        self.size = size
        self.transform_target = transforms.Compose([transforms.Resize(size),
                                transforms.ToTensor()])
        
    def __call__(self,index,dataframe,dic,label_folder):
        names = dataframe.iloc[index,1:].values.tolist()
        image = torch.zeros((self.size[0],self.size[1]))
        for i,j in enumerate(names):
            if j != 'NAN':
                img = Image.open('{0}\{1}\{2}'.format(dic,label_folder,j))
                img_a = self.transform_target(img).max(axis = 0)[0]
                image[img_a == 1] = i + 1
            else:
                continue
        return image
    
class Segdata(Dataset):

    def __init__(self,transform,image_size, csv_file, dic,input_label_foler):

        self.data = pd.read_csv('{0}\{1}'.format(dic,csv_file))
        self.dic = dic
        self.input_label_foler = input_label_foler
        self.transform = transform
        self.get_label = GET_LABEL(size = image_size)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, 'x_name']
        #print(img_name)
        image = Image.open(r'{0}\{1}\{2}'.format(self.dic,self.input_label_foler[0],img_name))
        image = self.transform(image)
        label = self.get_label(idx,self.data,self.dic,self.input_label_foler[1])
        return {'image': image,
                'labels': label
                }

    
    
def get_frequency(csv,image_size,num_class,dic,label_folder):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    label_sum = torch.zeros((csv.shape[0],image_size[0],image_size[1]))
    
    get_label = GET_LABEL(size = image_size)
    
    
    for i in range(csv.shape[0]):
        
        label_sum[i,:,:] = get_label(i,csv,dic,label_folder)
        
    class_frequency_list = torch.zeros((num_class,1),device = device)
    
    for i in range(num_class):
        
        class_frequency = (label_sum[:,:,:] == i).sum().numpy()/(csv.shape[0]*image_size[0]*image_size[1])
        
        class_frequency_list[i] = class_frequency
        
    return class_frequency_list    

def IoU(pred, target, n_classes):

    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(0, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            print(cls,float(intersection) / float(max(union, 1)))
    ious = [i for i in ious if np.isnan(i) == False]
    return np.mean(ious)

def IoU_batch(pred_batch, target_batch, n_classes):
    iou_batch = []
    for i in range(pred_batch.size()[0]):
        pred = torch.argmax(pred_batch[i], dim=0)
        ious = []
        pred = pred.view(-1)
        target = target_batch[i].view(-1)

        # Ignore IoU for background class ("0")
        for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
            pred_inds = pred == cls
            target_inds = target == cls

            intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
            union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
            if union == 0:
                ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
            else:
                ious.append(float(intersection) / float(max(union, 1)))
        ious_ = [i for i in ious if (np.isnan(i)==False)]
        
        iou_batch.append(np.mean(ious_))

    return np.mean(iou_batch)


def get_region(density,label,original_size,min_win=5,max_win = 10,stride = 1,thres=0.0001,scale = 0.05):
    featuremap = torch.argmax(density, dim=0)
    H = featuremap.size()[0]
    W = featuremap.size()[1]
    hwratio = H/W
    rec_list = []
    coor = []
    prob = []
    pix = featuremap[featuremap==label].size()
    image_size_feature = featuremap.size()
    x_ratio = original_size[0]/image_size_feature[1]
    y_ratio = original_size[1]/image_size_feature[0]
    size = []
    for i in range(min_win,max_win+1,3):
        size.append((i,i))
    
    for s in size:
        
        for y in range(H-s[1]+1):
            
            for x in range(W-s[0]+1):
                
                y1 = y*stride
                y2 = min(y*stride+s[1],H+1)
                x1 = x*stride
                x2 = min(x*stride+s[0],W+1)
                
                rec = featuremap[y1:y2,x1:x2]
                
                if label in rec:
                    summation = torch.exp(density[:,y1:y2,x1:x2]).sum(axis = 0)
                    single = torch.exp(density[label,y1:y2,x1:x2])
                    r = (single/summation).mean()
                    #print(rec[rec==5].size())
                    coor.append([x1,y1,x2,y2])
                    prob.append(r)
                    rec_list.append(rec)
                    
    res = torchvision.ops.nms(torch.tensor(coor,dtype = torch.float),torch.tensor(prob),thres)
    
    coordinates_list = []
    prob_list = []
    
    for index in res:
        col,row = torch.where(rec_list[index]==label)
        y_max = (col.float().max()+coor[index][1])*(1+scale)
        y_min = (col.float().min()+coor[index][1])*(1-scale)
        x_max = (row.float().max()+coor[index][0])*(1+scale)
        x_min = (row.float().min()+coor[index][0])*(1-scale)

        x1 = x_min
        y1 = y_min
        x2 = x_max
        y2 = y_max
        
        x1_ = int(x1*x_ratio)
        y1_ = int(y1*y_ratio)
        x2_ = int(x2*x_ratio)
        y2_ = int(y2*y_ratio)
        
        coordinates_list.append([x1_,y1_,x2_,y2_])
        prob_list.append(prob[index])
        
    return coordinates_list,prob_list

def get_region_v2(density,label,original_size,scale = 0.1,thres = 0.001):
    coor_list = []
    prob_list = []
    feature_map = torch.argmax(density, dim=0)
    feature_map[feature_map!=label] = 0
    feature_map[feature_map == label] = 1
    feature_map_array = feature_map.detach().cpu().numpy()
    x_ratio = original_size[0]/feature_map_array.shape[1]
    y_ratio = original_size[1]/feature_map_array.shape[0]

    _,index = np.unravel_index(np.where(feature_map_array==1),feature_map_array.shape)
    clustering = DBSCAN(eps=np.sqrt(feature_map_array.shape[0]*feature_map_array.shape[1])/20).fit(np.c_[index[0],index[1]])
    index_combined = np.c_[index[0],index[1],clustering.labels_]
    for i in np.unique(clustering.labels_):
        width = index_combined[index_combined[:,2]==i][:,1]
        height = index_combined[index_combined[:,2]==i][:,0]

        y_max = min(np.ceil((height.max()+1)*(1+scale)).astype(int),feature_map_array.shape[0])
        y_min = max(np.floor((height.min())*(1-scale)).astype(int),0)
        x_max = min(np.ceil((width.max()+1)*(1+scale)).astype(int),feature_map_array.shape[1])
        x_min = max(np.floor((width.min())*(1-scale)).astype(int),0)
        if (x_max-x_min)*(y_max-y_min)/(feature_map_array.shape[0]*feature_map_array.shape[1])>thres:

            x1 = max(x_min*x_ratio,0)
            y1 = max(y_min*y_ratio,0)
            x2 = min(x_max*x_ratio,original_size[0])
            y2 = min(y_max*y_ratio,original_size[1])
            coor_list.append([x1,y1,x2,y2])
            summation = torch.exp(density[:,y_min:y_max,x_min:x_max]).sum(axis = 0)
            single = torch.exp(density[label,y_min:y_max,x_min:x_max])
            r = (single/summation).mean()
            prob_list.append(r)
    return coor_list,prob_list

def decode_color(image,colors):
    
    if len(image.shape) <= 1:
        
        image = image.reshape(-1,1)
    
    img = np.zeros((image.shape[0],image.shape[1],3))
    
    for H in range(image.shape[0]):
        
        for W in range(image.shape[1]):
            
            for i in range(len(colors)+1):
                
                if image[H,W] == i:
                    
                    img[H,W,:] = colors[i]
                    
    return np.uint8(img)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def image_adjust(image):
    
    grayscale = np.array(image.convert('L'))
    median = np.median(grayscale)
    
    ratio = 130/median
    
    #image = adjust_gamma(np.array(image),gamma = ratio*0.1)
    #image = Image.fromarray(image)
    
    brightness = ImageEnhance.Brightness(image)
    contrast = ImageEnhance.Contrast(image)
    saturation = ImageEnhance.Color(image)
    
    image = saturation.enhance(ratio)
    image = contrast.enhance(ratio)
    image = brightness.enhance(ratio)

    
    return image

def cal_area(pred1,pred2):
    if len(pred1.shape)>2:
        pred1 = pred1.max(axis = 2)
    if len(pred2.shape)>2:
        pred2 = pred2.max(axis = 2)
    size1 = pred1.size
    size2 = pred2.size
    pred1[pred1!=0] = 1
    pred2[pred2!=0] = 1
    a = pred1.sum()/size1
    b = pred2.sum()/size2

    print(str(a*100)+'%')
    print(str(b*100)+'%')
    print(str(np.abs(a-b)/((a+b)/2)*100)+'%')
    return a,b

def get_metadata(image):
    import PIL.ExifTags
    exif = { PIL.ExifTags.TAGS[k]: v for k, v in image._getexif().items() if k in PIL.ExifTags.TAGS }
    return exif

def train(model, iterator, optimizer, criterion,loss_list,num_c):
    loss_ = []
    iou = []

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.train()
    
    for i in iterator:
        
        input = i['image'].to(device)
        target = i['labels'].to(device)
        res = model(input)
        loss = criterion(res,target.long())
        loss_.append(loss.item()) 
        iou.append(IoU_batch(res,target,num_c))
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        optimizer.zero_grad()
        
    return np.mean(loss_),np.mean(iou)

def evaluate(model, iterator, criterion,num_c):
    loss_ = []
    iou = []

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    with torch.no_grad():
    
        for i in iterator:

            input = i['image'].to(device)
            target = i['labels'].to(device)
            res = model(input)
            loss = criterion(res,target.long())
            loss_.append(loss.item()) 
            iou.append(IoU_batch(res,target,num_c))

        
    return np.mean(loss_),np.mean(iou)

def predict(image,transform_deeplab,dlab2,dlab2_detail,label,original_size,thres=0.001,scale = 0.05):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    validation = transform_deeplab(image).unsqueeze(0).to(device)

    coordinates,prob = get_region_v2(dlab2(validation,interpolate = False).squeeze(),
                                  label = label,original_size = original_size,
                                  thres = thres,scale = scale)

    coor_list = []
    mask_list = []
    om_list = []
    prob_list = []
    filtered_co = []
    for i,co in enumerate(coordinates):

        num = i

        x1 = co[0]
        x2 = co[2]
        y1 = co[1]
        y2 = co[3]

        if 0 not in np.array(image)[int(y1):int(y2),int(x1):int(x2)].shape:

            coor_list.append([(int(x1),int(y1)),(int(x2),int(y2))])
            filtered_co.append(co)
            prob_list.append(prob[i])

            frcnn_cropped = Image.fromarray(np.array(image)[int(y1):int(y2),int(x1):int(x2)])

            original_size = frcnn_cropped.size

            val = transform_deeplab(frcnn_cropped).unsqueeze(0).to(device)

            res_test_2 = dlab2_detail(val)

            res_test = F.interpolate(res_test_2, size=(original_size[1],
                                                      original_size[0]), mode='bilinear', align_corners=False)

            torch.cuda.empty_cache()

            om = torch.argmax(res_test.squeeze(), dim=0).detach().cpu().numpy()
            om_list.append(om)
            #mask_list.append(decode_color(om,colors))


    #rec = np.array(image)
    #mask_image = np.array(image)
    binary_image = np.zeros(np.array(image).shape[:2])
    for mask,i in enumerate(coor_list):
        #rec = cv2.rectangle(rec,i[0],i[1],(255, 0, 0),2)

        #mask_image[coor_list[mask][0][1]:coor_list[mask][1][1],
        #coor_list[mask][0][0]:coor_list[mask][1][0],:] = mask_list[mask]

        binary_image[coor_list[mask][0][1]:coor_list[mask][1][1],
        coor_list[mask][0][0]:coor_list[mask][1][0]] = om_list[mask]
        
    return binary_image
