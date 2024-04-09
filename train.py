import einops
from torch.utils.data import DataLoader
import torch.nn as nn
import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader
from collections import OrderedDict
import imageio
from skimage import metrics
from utils.func_pfm import *
import cv2
from tensorboardX import SummaryWriter
from utils.inference_method import test_m1 as test

def main(args):

    ''' Create Dir for Save'''
    log_dir, checkpoints_dir, val_dir = create_dir(args)

    ''' Logger '''
    logger = Logger(log_dir, args)
    
    ''' Tensorboard '''
    writer = SummaryWriter(log_dir="{}logs/{}".format(log_dir, args.save_prefix),
                        comment="Training curve for LFRRN")

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA Training LOADING '''
    logger.log_string('\nLoad Training Dataset ...')
    train_Dataset = TrainSetDataLoader(args)
    logger.log_string("The number of training data is: %d" % len(train_Dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_Dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=True,)

    ''' DATA Validation LOADING '''
    logger.log_string('\nLoad Validation Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    logger.log_string("The number of validation data is: %d" % length_of_tests)


    ''' MODEL LOADING '''
    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)
    net = net.to(device)
    cudnn.benchmark = True

    ''' Set '''
    if args.MGPU == 4:
        net = nn.DataParallel(net, device_ids=[0,1,2,3])
        logger.log_string('\nusing 4 GPUs ...')
    elif args.MGPU == 2:
        net = nn.DataParallel(net, device_ids=[0,1])
        logger.log_string('\nusing 2 GPUs ...')
    else:
        net = nn.DataParallel(net)
        logger.log_string('\nusing 1 GPU ...')

    ''' Print Parameters '''
    logger.log_string('\nPARAMETER ...')
    logger.log_string(args)

    ''' LOSS LOADING '''
    criterion = MODEL.get_loss(args).to(device)

    ''' Optimizer '''
    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad == True],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    ''' resume '''
    if args.resume:
        resume_path = args.resume
        if os.path.isfile(resume_path):
            logger.log_string("\n==> loading checkpoint {} for resume".format(resume_path))
            checkpoint = torch.load(resume_path)
            net.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
        else:
            logger.log_string("\n==> no model found at {}".format(args.resume))
            start_epoch = 0
    else:
        start_epoch = 0

    ''' retrain '''
    if args.retrain:
        retrain_path = args.retrain
        if os.path.isfile(retrain_path):
            logger.log_string("\n==> loading checkpoint {} for retrain".format(retrain_path))
            checkpoint = torch.load(retrain_path)
            net.load_state_dict(checkpoint['model'])
        else:
            logger.log_string("\n==> no model found at '{}'".format(retrain_path))
    else:
        logger.log_string("\n==> This is not a retrained experiment")

    ''' TRAINING & TEST '''
    logger.log_string('\nStart training...')

    for idx_epoch in range(start_epoch, args.epoch):
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))

        ''' Training '''
        loss_epoch_train = train(train_loader, device, net, criterion, optimizer, idx_epoch + 1, writer)
        logger.log_string('The %dth Train, loss is: %.5f, psnr is %.5f, ssim is %.5f' %
                          (idx_epoch + 1, loss_epoch_train, 0.00, 0.00))
        writer.add_scalar("train/recon_loss", loss_epoch_train, idx_epoch + 1)

        ''' Save PTH  '''
        if args.local_rank == 0:
            save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_epoch_%02d_model.pth' % (
            args.model_name, args.angRes_in, args.angRes_in, idx_epoch + 1)
            state = {
                'epoch': idx_epoch + 1,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, save_ckpt_path)
            logger.log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))

        ''' Validation '''
        step = 20
        if (idx_epoch + 1)%step == 0 or idx_epoch > args.epoch-step:
            with torch.no_grad():
                ''' Create Excel for PSNR/SSIM '''

                psnr_testset = []
                ssim_testset = []
                for index, test_name in enumerate(test_Names):
                    test_loader = test_Loaders[index]

                    epoch_dir = val_dir.joinpath('VAL_epoch_%02d' % (idx_epoch + 1))
                    epoch_dir.mkdir(exist_ok=True)
                    save_dir = epoch_dir.joinpath(test_name)
                    save_dir.mkdir(exist_ok=True)

                    psnr_iter_test, ssim_iter_test, LF_name = test(test_loader, device, net, save_dir)

                    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
                    psnr_testset.append(psnr_epoch_test)
                    ssim_epoch_test = float(np.array(ssim_iter_test).mean())
                    ssim_testset.append(ssim_epoch_test)
                    logger.log_string('The %dth Test on %s, psnr/ssim is %.2f/%.4f' % (
                    idx_epoch + 1, test_name, psnr_epoch_test, ssim_epoch_test))
                    pass
                psnr_mean_test = float(np.array(psnr_testset).mean())
                ssim_mean_test = float(np.array(ssim_testset).mean())
                logger.log_string('The mean psnr/ssim on testsets is %.5f/%.5f' % (psnr_mean_test,ssim_mean_test))
                pass
            pass

        ''' scheduler '''
        scheduler.step()
        pass
    pass


def train(train_loader, device, net, criterion, optimizer, epoch, writer):
    ''' training one epoch '''
    psnr_iter_train = []

    loss_iter_train = []
    ssim_iter_train = []
    angres = 5
    ref_pos = angres * angres // 2 + angres // 2
    for idx_iter, (data, label, data_info) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        data = data.to(device)
        label = label.to(device) 
        Disparity, out = net(data, data_info)
        loss = criterion(Disparity, out, label, data_info)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        loss_iter_train.append(loss.data.cpu())

        if idx_iter % 5 == 0:
            print("{}: Epoch {}, [{}/{}]: SR loss: {:.10f}".format(2023, epoch, idx_iter, len(train_loader),
                                                               loss.cpu().data))
            writer.add_scalar("train/recon_loss_iter", loss.cpu().data, idx_iter + (epoch - 1) * len(train_loader))

        pass

    loss_epoch_train = float(np.array(loss_iter_train).mean())

    return loss_epoch_train

if __name__ == '__main__':
    from option import args
    args.path_log = args.path_log + args.save_prefix
    main(args)
