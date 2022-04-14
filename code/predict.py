#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   predict.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
4/13/22 9:55 AM   tango      1.0         None
"""
import argparse
import os
import random
import json

from dataloaders.dataset import OwnBaseDataSets
import numpy as np
import torch
from medpy import metric
from networks.net_factory import net_factory
from torch.utils.data import DataLoader
from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/throat', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='throat/Uncertainty_Rectified_Pyramid_Consistency', help='experiment_name')
parser.add_argument('--method', type=str,
                    default='Uncertainty_Rectified_Pyramid_Consistency', help='method_name')
parser.add_argument('--in_chns', type=int,
                    default=3, help='in_chns')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=5,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=20,
                    help='labeled data')
parser.add_argument('--patch_size', type=list,  default=[512,512],
                    help='patch size of network input')

args = parser.parse_args()
model_info=json.load(open("setting.json","r",encoding="utf8"))

smooth = 1e-6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if gt.sum()>0 and pred.sum()<=0:
        return 0,0
    if gt.sum()<=0 and pred.sum() > 0:
        return smooth,smooth
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt)
        jc=metric.binary.jc(pred, gt)
        return dice, jc
    else:
        return 0,0

def throat_single_volume(image, label, net, classes, patch_size=512):
    image, label = image.cpu().detach().numpy(), label.cpu().detach().numpy() # 1 c w h  label 1 w h
    input= torch.from_numpy(image).float().to(device)
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(
            net(input), dim=1), dim=1)
        out = out.cpu().detach().numpy() # 1 w h
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            out == i, label == i))
    return metric_list,out

def throat_single_volume_ds(image, label, net, classes, patch_size=512):
    image, label = image.cpu().detach().numpy(), label.cpu().detach().numpy() # 1 c w h  label 1 w h
    input= torch.from_numpy(image).float().to(device)
    net.eval()
    with torch.no_grad():
        output_main, _, _, _ = net(input)
        out = torch.argmax(torch.softmax(
            output_main, dim=1), dim=1)
        out = out.cpu().detach().numpy() # 1 w h
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            out == i, label == i))
    return metric_list,out

def make_image(tensor,idx,snapshot_path):
    from PIL import Image
    image = Image.fromarray((tensor[0]*50).astype(np.uint8))
    image.save(os.path.join(snapshot_path,idx+".png"), format='PNG')

def metrics_to_file(metric_list,performance,snapshot_path):
    info={
        "categories":metric_list.tolist(),
        "performance":performance.tolist()
    }
    s=json.dumps(info,indent=2)
    with open(os.path.join(snapshot_path,"performance.txt"),"a") as f:
        f.write(s)

def val(args,snapshot_path):
    num_classes = args.num_classes
    in_chns=args.in_chns
    method=args.method
    db_val = OwnBaseDataSets(base_dir=args.root_path, split="val", output_size=args.patch_size)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers=1)

    model = net_factory(net_type=model_info[method]['model'][0], in_chns=in_chns, class_num=num_classes)
    # load weight
    model.load_state_dict(torch.load(model_info[method]['weights'][0], map_location=device), strict=False)
    model.eval()
    metric_list = []
    print("Process model")
    for i_batch, sampled_batch in enumerate(valloader):
        if not model_info[method]["ds"]:
            metric_i,out = throat_single_volume(
                sampled_batch["image"].to(device), sampled_batch["label"].to(device), model, classes=num_classes, patch_size=args.patch_size)
        else:
            metric_i,out = throat_single_volume_ds(
                sampled_batch["image"].to(device), sampled_batch["label"].to(device), model, classes=num_classes, patch_size=args.patch_size)
        metric_list.append(metric_i)
        #plot
        make_image(out,sampled_batch['idx'][0],snapshot_path)
        print(f"\t\tprocessing {sampled_batch['idx']}")
    metric_list = np.array(metric_list)
    metric_list = np.sum(metric_list, axis=0) / (np.sum(metric_list != 0, axis=0) + 1e-5)
    performance = np.mean(metric_list, axis=0)
    metrics_to_file(metric_list,performance,snapshot_path)

    if len(model_info[method]['model'])>1:
        print("Process model2")
        if model_info[method]['model'][1]=="vit":
            config = get_config(args)
            model2=ViT_seg(config, img_size=args.patch_size,
                           num_classes=args.num_classes).to(device)
        else:
            model2 = net_factory(net_type=model_info[method]['model'][1], in_chns=in_chns, class_num=num_classes)
        #load weight\
        model2.load_state_dict(torch.load(model_info[method]['weights'][1], map_location=device), strict=False)
        model2.eval()
        metric_list = []
        for i_batch, sampled_batch in enumerate(valloader):
            if not model_info[method]["ds"]:
                metric_i, out = throat_single_volume(
                    sampled_batch["image"].to(device), sampled_batch["label"].to(device), model2, classes=num_classes,
                    patch_size=args.patch_size)
            else:
                metric_i, out = throat_single_volume_ds(
                    sampled_batch["image"].to(device), sampled_batch["label"].to(device), model2, classes=num_classes,
                    patch_size=args.patch_size)
            metric_list.append(metric_i)
            # plot
            make_image(out, "model2_"+sampled_batch['idx'][0], snapshot_path)
            print(f"\t\tprocessing {sampled_batch['idx']}")
        metric_list = np.array(metric_list)
        metric_list = np.sum(metric_list, axis=0) / (np.sum(metric_list != 0, axis=0) + 1e-5)
        performance = np.mean(metric_list, axis=0)
        metrics_to_file(metric_list, performance, snapshot_path)

if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    snapshot_path = "../model/{}_{}_labeled/figure/".format(
        args.exp, args.labeled_num)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    val(args,snapshot_path)