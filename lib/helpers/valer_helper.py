import os
import tqdm
import json

import torch
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.helpers.utils_helper import calc_iou
import time
import numpy as np
from utils import misc
import json

class Tester(object):
    def __init__(self, cfg, model, dataloader, logger,loss, train_cfg=None, model_name='mono3dvg'):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'Mono3DRefer')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.model_name = model_name
        self.mono3dvg_loss = loss

    def test(self):
        # test a checkpoint
        checkpoint_path = os.path.join(self.output_dir, self.cfg['pretrain_model'])
        assert os.path.exists(checkpoint_path)
        load_checkpoint(model=self.model,
                        optimizer=None,
                        filename=checkpoint_path,
                        map_location=self.device,
                        logger=self.logger)
        self.model.to(self.device)

        self.inference()

    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params_in_millions = total_params / 1e6
        print(f"模型推理所需的参数总量: {total_params_in_millions:.2f} M")
        # print(f'模型参数量: {param_count}')
        self.logger.info(f"模型推理所需的参数总量: {total_params_in_millions:.2f} M")
        # pp

        
        with open("/sdc1/songcl/mono3D/data/mono3DRefer/test_instanceID_split.json", "r") as file:
            test_instanceID_spilt = json.load(file)
        results = {}
        iou_3dbox_test = {"Unique": [], "Multiple": [], "Overall": [], "Near": [], "Medium": [], "Far": [],
                          "Easy": [], "Moderate": [], "Hard": []}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0
        json_dicts = {}
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)
            gt_3dboxes = info['gt_3dbox']
            infos = info
            gt_3dboxes = [[float(gt_3dboxes[j][i].detach().cpu().numpy()) for j in range(len(gt_3dboxes)) ] for i in range(len(gt_3dboxes[0]))]
            # [object.h,object.w,object.l,np.float32(object.pos[0]),np.float32(object.pos[1]),np.float32(object.pos[2])]
            captions = targets["text"]
            im_name = targets['image_id']
            instanceID = targets['instance_id']
            ann_id = targets['anno_id']
            batch_size = inputs.shape[0]

            start_time = time.time()
            outputs = self.model(inputs, calibs,  img_sizes, captions, im_name, instanceID, ann_id)
            end_time = time.time()
            model_infer_time += end_time - start_time

            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg['topk'])

            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items() if key not in ['gt_3dbox']}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets, pred_3dboxes,rl = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,type='extra')

            for i in range(batch_size):
                
                iID = instanceID[i]
                pre_box3D = np.array(pred_3dboxes[i])
                pre_box3D = np.array([pre_box3D[3], pre_box3D[4], pre_box3D[5], pre_box3D[0],pre_box3D[1], pre_box3D[2]], dtype=np.float32)
                gt_box3D = np.array(gt_3dboxes[i])
                gt_box3D_save = gt_box3D.copy()
                gt_box3D = np.array([gt_box3D[3], gt_box3D[4], gt_box3D[5], gt_box3D[0], gt_box3D[1], gt_box3D[2]],dtype=np.float32)
                ### x y z w h l
                
                gt_box3D[1] -= gt_box3D[3] / 2    # real 3D center in 3D space

                pre_box3D[1] -= pre_box3D[3] / 2  # real 3D center in 3D space
                IoU = calc_iou(pre_box3D, gt_box3D)

                ### 将结果保存下来 
                key_name = im_name[i] +'_'+ str(info['instance_id'][i])+'_'+ str(info['anno_id'][i])
                if json_dicts.get(key_name,None) is not None:
                    print(f'key:{key_name} retry')
                    break
                result_dict = {}
                result_dict['im_name'] = im_name[i]
                result_dict['instanceID'] = int(instanceID[i])
                result_dict['ann_id'] = int(ann_id[i])
                result_dict['label_2_pre'] = str(rl[i])
                result_dict['label_2_gt'] = gt_box3D_save.tolist()
                result_dict['IoU'] = IoU
                result_dict['description'] = captions[i]
                json_dicts[key_name]=result_dict

                # file_path = os.path.join(self.output_dir, 'result.json')
                
                # try:
                #     # 打开文件以写入模式
                #     with open(file_path, 'w') as json_file:
                #         # 使用 json.dump 将列表写入文件
                #         json.dump(json_dicts, json_file,indent=2)
                #     print(f"列表已成功保存到 {file_path}")
                # except Exception as e:
                #     print(f"保存列表到文件时出现错误: {e}")
        
                # pp
                iou_3dbox_test["Overall"].append(IoU)
                if iID in test_instanceID_spilt['Unique']:
                    iou_3dbox_test["Unique"].append(IoU)
                else:
                    iou_3dbox_test["Multiple"].append(IoU)

                if iID in test_instanceID_spilt['Near']:
                    iou_3dbox_test["Near"].append(IoU)
                elif iID in test_instanceID_spilt['Medium']:
                    iou_3dbox_test["Medium"].append(IoU)
                elif iID in test_instanceID_spilt['Far']:
                    iou_3dbox_test["Far"].append(IoU)

                if iID in test_instanceID_spilt['Easy']:
                    iou_3dbox_test["Easy"].append(IoU)
                elif iID in test_instanceID_spilt['Moderate']:
                    iou_3dbox_test["Moderate"].append(IoU)
                elif iID in test_instanceID_spilt['Hard']:
                    iou_3dbox_test["Hard"].append(IoU)

            results.update(dets)
            progress_bar.update()

            if batch_idx % 30 == 0:
                acc5 = np.sum(np.array((np.array(iou_3dbox_test["Overall"]) > 0.5), dtype=float)) / len(iou_3dbox_test["Overall"])
                acc25 = np.sum(np.array((np.array(iou_3dbox_test["Overall"]) > 0.25), dtype=float)) / len(iou_3dbox_test["Overall"])
                miou = sum(iou_3dbox_test["Overall"]) / len(iou_3dbox_test["Overall"])

                print_str ='Epoch: [{}/{}]\t' \
                           'Accu25 {acc25:.2f}%\t' \
                           'Accu5 {acc5:.2f}%\t' \
                           'Mean_iou {miou:.2f}%\t' \
                    .format(
                    batch_idx, len(self.dataloader), \
                    acc25=acc25 * 100, acc5=acc5 * 100, miou=miou * 100
                )
                # print(print_str)
                self.logger.info(print_str)

        print("inference on {} objects, inference time {} ".format(len(self.dataloader), model_infer_time/len(self.dataloader)))
        self.logger.info("inference on {} objects, inference time {} ".format(len(self.dataloader), model_infer_time/len(self.dataloader)))

        progress_bar.close()

        # save the result for evaluation.
        self.logger.info('==> Mono3DVG Evaluation ...')

        for split_name, iou_3dbox in iou_3dbox_test.items():
            self.logger.info("------------" + split_name + "------------")
            acc5 = np.sum(np.array((np.array(iou_3dbox) > 0.5), dtype=float)) / len(iou_3dbox)
            acc25 = np.sum(np.array((np.array(iou_3dbox) > 0.25), dtype=float)) / len(iou_3dbox)
            miou = sum(iou_3dbox) / len(iou_3dbox)
            print_str = 'Accu25 {acc25:.2f}%\t' \
                        'Accu5 {acc5:.2f}%\t' \
                        'Mean_iou {miou:.2f}%\t' \
                .format(
                acc25=acc25 * 100, acc5=acc5 * 100, miou=miou * 100
            )
            # print(print_str)
            self.logger.info(print_str)

        

        # 示例列表
        # my_list = [1, 2, 3, 4, 5]

        # 定义要保存的 JSON 文件路径
        file_path = os.path.join(self.output_dir, 'result.json')
        
        try:
            # 打开文件以写入模式
            with open(file_path, 'w') as json_file:
                # 使用 json.dump 将列表写入文件
                json.dump(json_dicts, json_file,indent=2)
            print(f"列表已成功保存到 {file_path}")
        except Exception as e:
            print(f"保存列表到文件时出现错误: {e}")


    def evaluate(self, epoch):
        torch.set_grad_enabled(False)
        self.model.eval()

        iou_3dbox = []
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)
            gt_3dboxes = info['gt_3dbox']
            gt_3dboxes = [[float(gt_3dboxes[j][i].detach().cpu().numpy()) for j in range(len(gt_3dboxes)) ] for i in range(len(gt_3dboxes[0]))]

            captions = targets["text"]
            im_name = targets['image_id']
            instanceID = targets['instance_id']
            ann_id = targets['anno_id']
            batch_size = inputs.shape[0]

            for key in targets.keys():
                if key not in ['image_id', 'text']:
                    targets[key] = targets[key].to(self.device)
            targets = self.prepare_targets(targets, batch_size)

            outputs = self.model(inputs, calibs,  img_sizes, captions, im_name, instanceID, ann_id)

            # compute Loss
            mono3dvg_losses_dict = self.mono3dvg_loss(outputs, targets)
            weight_dict = self.mono3dvg_loss.weight_dict
            mono3dvg_losses_dict = misc.reduce_dict(mono3dvg_losses_dict)
            mono3dvg_losses_dict_log = {}
            mono3dvg_losses_log = 0
            for k in mono3dvg_losses_dict.keys():
                if k in weight_dict:
                    mono3dvg_losses_dict_log[k] = (mono3dvg_losses_dict[k] * weight_dict[k]).item()
                    mono3dvg_losses_log += mono3dvg_losses_dict_log[k]
            mono3dvg_losses_dict_log["loss_mono3dvg"] = mono3dvg_losses_log

            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg['topk'])
            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items() if key not in ['gt_3dbox']}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets, pred_3dboxes = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,)

            for i in range(batch_size):
                pre_box3D = np.array(pred_3dboxes[i])
                pre_box3D = np.array([pre_box3D[3], pre_box3D[4], pre_box3D[5], pre_box3D[0], pre_box3D[1], pre_box3D[2]],dtype=np.float32)
                gt_box3D = np.array(gt_3dboxes[i])
                gt_box3D = np.array([gt_box3D[3], gt_box3D[4], gt_box3D[5], gt_box3D[0], gt_box3D[1], gt_box3D[2]],dtype=np.float32)

                gt_box3D[1] -= gt_box3D[3] / 2    # real 3D center in 3D space
                pre_box3D[1] -= pre_box3D[3] / 2  # real 3D center in 3D space
                IoU = calc_iou(pre_box3D, gt_box3D)

                iou_3dbox.append(IoU)

            if batch_idx % 30 == 0:
                acc5 = np.sum(np.array((np.array(iou_3dbox) > 0.5), dtype=float)) / len(iou_3dbox)
                acc25 = np.sum(np.array((np.array(iou_3dbox) > 0.25), dtype=float)) / len(iou_3dbox)
                miou = sum(iou_3dbox) / len(iou_3dbox)

                print_str ='Evaluation: [{}][{}/{}]\t' \
                           'Loss_mono3dvg: {:.2f}\t' \
                           'Accu25 {:.2f}%\t' \
                           'Accu5 {:.2f}%\t' \
                           'Mean_iou {:.2f}%\t' \
                    .format(
                    epoch, batch_idx, len(self.dataloader), \
                    mono3dvg_losses_dict_log["loss_mono3dvg"], \
                    acc25 * 100, acc5 * 100,\
                    miou * 100
                )
                # print(print_str)
                self.logger.info(print_str)

        acc5 = np.sum(np.array((np.array(iou_3dbox) > 0.5), dtype=float)) / len(iou_3dbox)
        acc25 = np.sum(np.array((np.array(iou_3dbox) > 0.25), dtype=float)) / len(iou_3dbox)
        miou = sum(iou_3dbox) / len(iou_3dbox)
        print_str = 'Loss_mono3dvg: {:.2f}\t' \
                    'Accu25 {:.2f}%\t' \
                    'Accu5 {:.2f}%\t' \
                    'Mean_iou {:.2f}%\t' \
            .format(
            mono3dvg_losses_dict_log["loss_mono3dvg"],
            acc25 * 100, acc5 * 100,
            miou * 100
        )
        # print("Final Evaluation Result: ",print_str)
        self.logger.info("Final Evaluation Result: "+ print_str)
        return acc25* 100, acc5* 100, mono3dvg_losses_dict_log["loss_mono3dvg"]

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d','target_visual_classifier','target_depth_classifier']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    if key == 'target_visual_classifier' or key =='target_depth_classifier':
                        target_dict[key] = val[bz]
                    else:
                        target_dict[key] = val[bz][mask[bz]]
            targets_list.append(target_dict)
        return targets_list



