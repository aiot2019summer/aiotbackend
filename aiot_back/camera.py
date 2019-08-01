from __future__ import division
import time
from base_camera import BaseCamera
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import pickle as pkl
import argparse
from math import ceil
import paho.mqtt.client as mqtt

def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img, classes, point):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    if classes[cls]=='person':
        color = (0,0,150)
        cv2.rectangle(img, c1, c2,color, 1)
        #Add coordinates to the list
        point.append({"x":int(ceil(c2[0].item()+c1[0].item()/2)),"y":int(c2[1].item())})
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "t.avi", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "320", type = str)
    return parser.parse_args()

def on_connect(client, userdata, flags, rc):
    print("Connection returned with result code:" + str(rc))

def on_message(client, userdata, msg):
    print("Received message, topic:" + msg.topic + "payload:" + str(msg.payload))


class Camera(BaseCamera):
    """An emulated camera implementation that streams a repeated sequence of files."""

    @staticmethod
    def frames():
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect("127.0.0.1", 1883)
        client.loop_start()
        result, mid = client.subscribe("hello", 0)

        args = arg_parse()
        confidence = float(args.confidence)
        nms_thesh = float(args.nms_thresh)
        start = 0

        CUDA = torch.cuda.is_available()

        num_classes = 80

        CUDA = torch.cuda.is_available()
        
        bbox_attrs = 5 + num_classes
        
        print("Loading network.....")
        model = Darknet(args.cfgfile)
        model.load_weights(args.weightsfile)
        print("Network successfully loaded")

        model.net_info["height"] = args.reso
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0 
        assert inp_dim > 32

        if CUDA:
            model.cuda()
            
        model(get_test_input(inp_dim, CUDA), CUDA)

        model.eval()
        
        videofile = args.video
        
        cap = cv2.VideoCapture(videofile)

        assert cap.isOpened(), 'Cannot capture source'
        
        frames = 0
        start = time.time() 
        classes = load_classes('data/coco.names')

        while cap.isOpened():

            point=[]
            cap.set(1,frames)
            ret, frame = cap.read(frames)
            if ret:
                

                img, orig_im, dim = prep_image(frame, inp_dim)
                
                im_dim = torch.FloatTensor(dim).repeat(1,2)                        
                
                
                if CUDA:
                    im_dim = im_dim.cuda()
                    img = img.cuda()
                
                with torch.no_grad():   
                    output = model(Variable(img), CUDA)
                output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)              

                im_dim = im_dim.repeat(output.size(0), 1)
                scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
                
                output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
                output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
                
                output[:,1:5] /= scaling_factor
        
                for i in range(output.shape[0]):
                    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
                
                list(map(lambda x: write(x, orig_im, classes, point), output))
                ret, jpeg=cv2.imencode(".jpg", orig_im)
                yield jpeg.tobytes()
                time_now=time.time()-start 
                frames = int(30*time_now)
                mydict = str(point)[0:-1]+',{"time":'+str(time_now)+'}]'
                client.publish("hello", payload = mydict)
            else:
                break