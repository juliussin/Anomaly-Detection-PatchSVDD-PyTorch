from comet_ml import Experiment
import os
import argparse
import torch
from torchvision import transforms
from codes.augmentation import *
from codes import mvtecad
from functools import reduce
from torch.utils.data import DataLoader
from codes.datasets import *
from codes.networks import *
from codes.inspection import eval_encoder_NN_multiK
from codes.utils import *
from tqdm import tqdm


def train(args, experiment):
    obj = args.obj
    D = args.D
    lr = args.lr
        
    with task('Networks'):
        enc = EncoderHier(64, D).cuda()
        cls_64 = PositionClassifier(64, D).cuda()
        cls_32 = PositionClassifier(32, D).cuda()

        modules = [enc, cls_64, cls_32]
        params = [list(module.parameters()) for module in modules]
        params = reduce(lambda x, y: x + y, params)

        opt = torch.optim.Adam(params=params, lr=lr)

    with task('Datasets'):
        train_x = mvtecad.get_x_standardized(obj, mode='train')
        train_x = NHWC2NCHW(train_x)

        transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.2),
            AddGaussianNoise(0.1, 0.08)
        ])

        rep = 100
        datasets = dict()
        datasets[f'pos_64'] = PositionDataset(train_x, K=64, repeat=rep)
        datasets[f'pos_32'] = PositionDataset(train_x, K=32, repeat=rep)
        
        datasets[f'svdd_64'] = SVDD_Dataset(train_x, K=64, repeat=rep)
        datasets[f'svdd_32'] = SVDD_Dataset(train_x, K=32, repeat=rep)

        dataset = DictionaryConcatDataset(datasets)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    print('Start training')
    with experiment.train():
        for i_epoch in tqdm(range(args.epochs)):
            if i_epoch != 0:
                print(i_epoch)
                for module in modules:
                    module.train()
                for d in loader:
                    d = to_device(d, 'cuda', non_blocking=True)
                    opt.zero_grad()
                    loss_pos_64 = PositionClassifier.infer(cls_64, enc, d['pos_64'])
                    loss_pos_32 = PositionClassifier.infer(cls_32, enc.enc, d['pos_32'])
                    loss_svdd_64 = SVDD_Dataset.infer(enc, d['svdd_64'])
                    loss_svdd_32 = SVDD_Dataset.infer(enc.enc, d['svdd_32'])

                    loss = loss_pos_64 + loss_pos_32 + args.lambda_value * (loss_svdd_64 + loss_svdd_32)
                    
                    loss.backward()
                    opt.step()
                print("Loss: ", loss)
                experiment.log_metric("Loss", loss, step=i_epoch)
                experiment.log_metric("loss_pos_64", loss_pos_64, step=i_epoch)
                experiment.log_metric("loss_pos_32", loss_pos_32, step=i_epoch)
                experiment.log_metric("loss_svdd_64", loss_svdd_64, step=i_epoch)
                experiment.log_metric("loss_svdd_32", loss_svdd_32, step=i_epoch)
                

            if i_epoch % args.eval_interval == 0 or i_epoch == 0:
                aurocs = eval_encoder_NN_multiK(enc, obj)
                log_result(obj, aurocs, i_epoch, experiment)
                enc.save(obj, i_epoch)


def log_result(obj, aurocs, i_epoch, experiment):
    det_64 = aurocs['det_64'] * 100
    seg_64 = aurocs['seg_64'] * 100

    det_32 = aurocs['det_32'] * 100
    seg_32 = aurocs['seg_32'] * 100

    det_sum = aurocs['det_sum'] * 100
    seg_sum = aurocs['seg_sum'] * 100

    det_mult = aurocs['det_mult'] * 100
    seg_mult = aurocs['seg_mult'] * 100

    print(f'|K64| Det: {det_64:4.1f} Seg: {seg_64:4.1f} |K32| Det: {det_32:4.1f} Seg: {seg_32:4.1f} |mult| Det: {det_sum:4.1f} Seg: {seg_sum:4.1f} |mult| Det: {det_mult:4.1f} Seg: {seg_mult:4.1f} ({obj})')
    experiment.log_metric("det_64", det_64, step=i_epoch)
    experiment.log_metric("seg_64", seg_64, step=i_epoch)
    experiment.log_metric("det_32", det_32, step=i_epoch)
    experiment.log_metric("seg_32", seg_32, step=i_epoch)
    experiment.log_metric("det_sum", det_sum, step=i_epoch)
    experiment.log_metric("seg_sum", seg_sum, step=i_epoch)
    experiment.log_metric("det_mult", det_mult, step=i_epoch)
    experiment.log_metric("seg_mult", seg_mult, step=i_epoch)



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--obj', default='hazelnut', type=str)
    parser.add_argument('--lambda_value', default=1, type=float)
    parser.add_argument('--D', default=64, type=int)

    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--eval_interval', default=20, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--dataset', default="./dataset/mvtec_anomaly_detection/", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    ## 1. Parse Args
    args = parse_args()
    os.environ['DATASET_PATH'] = args.dataset
    mvtecad.set_root_path(args.dataset)

    ## 2. Setup Comet ML
    experiment = Experiment(api_key="cflnxIlb8WLD8q5235GDSxyXA", project_name="patch-svdd", workspace="kelvinliu04",)
    experiment.log_parameters(args)

    ## 3. Train
    train(args, experiment)
