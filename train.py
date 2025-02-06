import os
import torch
from models import models as networks
from models.models_CNN import resnet18, ResNet, BasicBlock
from modules import DTW_align, GreedyCTCDecoder, AttrDict, RMSELoss, save_checkpoint
from modules import mel2wav_vocoder, perform_STT
from utils import data_denorm, word_index
import torch.nn as nn
import torch.nn.functional as F
from NeuroTalkDataset import myDataset
import time
import torch.optim.lr_scheduler
import numpy as np
import torchaudio
from torchmetrics import CharErrorRate
import json
import argparse
import wavio
from torch.utils.tensorboard import SummaryWriter
from models.ESN_py.config import parse_args
from models.mamba_config import mamba_args
from soft_dtw import SoftDTW

    
def train(args, train_loader, models, criterions, optimizers, epoch, trainValid=True, inference=False):
    '''
    :param args: general arguments
    :param train_loader: loaded for training/validation/test dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: losses
    '''
    (optimizer_g) = optimizers
    
    # switch to train mode
    assert type(models) == tuple, "More than two models should be inputed (generator and discriminator)"

    epoch_loss_g = []

    total_batches = len(train_loader)
    
    for i, (input, target, target_cl,  data_info) in enumerate(train_loader):    

        print("\rBatch [%5d / %5d]"%(i,total_batches), sep=' ', end='', flush=True)
        
        input = input.cuda()
        target = target.cuda()
        target_cl = target_cl.cuda()
        voice = torch.squeeze(voice,dim=-1).cuda()
        labels = torch.argmax(target_cl,dim=1)

        
        # extract unseen
        idx_seen=[]
        for j in range(len(labels)):
            idx_seen.append(j)
        
        
        input = input[idx_seen]

        target = target[idx_seen]
        target_cl = target_cl[idx_seen]
        voice = voice[idx_seen]
        labels = labels[idx_seen]
        data_info = [data_info[0][idx_seen],data_info[1][idx_seen]]
        
        
        # general training         
        if len(input) != 0:
            # train generator
            mel_out, e_loss_g = train_G(args, input, target, voice, labels,  
                                                 models, criterions, optimizer_g, 
                                                 data_info, 
                                                 trainValid)
            epoch_loss_g.append(e_loss_g)

    args.loss_g = sum(epoch_loss_g[:,0]) / len(epoch_loss_g[:,0])
    args.loss_g_ctc = sum(epoch_loss_g[:,1]) / len(epoch_loss_g[:,1])
    args.cer_gt = sum(epoch_acc_g[:,0]) / len(epoch_acc_g[:,0])
    args.cer_recon = sum(epoch_acc_g[:,1]) / len(epoch_acc_g[:,1])

    
    # tensorboard
    if trainValid:
        tag = 'train'
    else:
        tag = 'valid'
        
    if not inference:
        args.writer.add_scalar("Loss_G/{}".format(tag), args.loss_g, epoch)
        args.writer.add_scalar("CER/{}".format(tag), args.cer_recon, epoch)
        args.writer.add_scalar("Loss_G_ctc/{}".format(tag), args.loss_g_ctc, epoch)
        # args.writer.add_scalar("Loss_G_recon/{}".format(tag), args.loss_g_recon, epoch)
        # args.writer.add_scalar("Loss_G_dtw/{}".format(tag), args.loss_g_dtw, epoch)
        
        
        



    print('\n[%3d/%3d]  CER-gt: %.4f CER-recon: %.4f / g-RMSE: %.4f g-lossCTC: %.4f' 
          % (i, total_batches,
             args.cer_gt, args.cer_recon,
             args.loss_g_recon, args.loss_g_ctc))
        
        
    return (args.loss_g, args.loss_g_ctc, args.cer_gt, args.cer_recon)


def train_G(args, input, target,  labels, models, criterions, optimizer_g, data_info, trainValid):    
    (model_g, vocoder, model_STT, decoder_STT) = models
    (criterion_recon, criterion_dtw) =  criterions
    
    if trainValid:
        model_g.train()
    else:
        model_g.eval()

    
    # Adversarial ground truths 1:real, 0: fake
    valid = torch.ones((len(input), 1), dtype=torch.float32).cuda()
    
    ###############################
    # Train Generator
    ###############################
    
    if trainValid:
        for p in model_g.parameters():
            p.requires_grad_(True)   # unfreeze G

        # set zero grad    
        optimizer_g.zero_grad()
        
        # Run Generator
        output = model_g(input.permute(0,2,1))
    else:
        with torch.no_grad():
            # run generator
            output = model_g(input.permute(0,2,1))

    #print(f"output's shape: {output.shape}")
    # DTW
    mel_out = output.clone()
    
    # generator loss
    loss_recon = criterion_recon(mel_out, target)
    

    target_denorm = target
    output_denorm = mel_out
    
    #reshape
    output_reshape = output_denorm.reshape(-1,1,80,172)
    
    
    # DTW Loss
    loss_dtw = criterion_dtw(mel_out, target)
    loss_dtw = loss_dtw.mean()

    ###############################
    # Loss from Vocoder - STT
    ###############################
    
    gt_label=[]
    gt_label_idx=[]
    gt_length=[]
    for j in range(len(target)):
        gt_label.append(args.word_label[labels[j].item()])
        gt_label_idx.append(args.word_index[labels[j].item()])
        gt_length.append(args.word_length[labels[j].item()])
    gt_label_idx = torch.tensor(np.array(gt_label_idx),dtype=torch.int64)
    gt_length = torch.tensor(gt_length,dtype=torch.int64)
    
    # target
    ##### HiFi-GAN
    wav_target = vocoder(target_denorm)
    wav_target = torch.reshape(wav_target, (len(wav_target),wav_target.shape[-1]))
    
    #### resampling
    wav_target = torchaudio.functional.resample(wav_target, args.sample_rate_mel, args.sample_rate_STT)   
    if wav_target.shape[1] !=  voice.shape[1]:
        p = voice.shape[1] - wav_target.shape[1]
        p_s = p//2
        p_e = p-p_s
        wav_target = F.pad(wav_target, (p_s,p_e))
    
    # recon
    ##### HiFi-GAN
    wav_recon = vocoder(output_denorm)
    wav_recon = torch.reshape(wav_recon, (len(wav_recon),wav_recon.shape[-1]))
    
    #### resampling
    wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)   
    if wav_recon.shape[1] !=  voice.shape[1]:
        p = voice.shape[1] - wav_recon.shape[1]
        p_s = p//2
        p_e = p-p_s
        wav_recon = F.pad(wav_recon, (p_s,p_e))

    ##### STT Wav2Vec 2.0
    emission_gt, _ = model_STT(voice)
    emission_recon, _ = model_STT(wav_recon)
   
    # CTC loss
    input_lengths = torch.full(size=(emission_gt.size(dim=0),), fill_value=emission_gt.size(dim=1), dtype=torch.long)
    emission_recon_ = emission_recon.log_softmax(2)
    loss_ctc = criterion_ctc(emission_recon_.transpose(0, 1), gt_label_idx, input_lengths, gt_length) 

    
    # total generator loss
    loss_g = loss_recon

    # decoder STT
    transcript_gt = []
    transcript_recon = []

    for j in range(len(voice)):
        transcript = decoder_STT(emission_gt[j])   
        transcript_gt.append(transcript)
            
        transcript = decoder_STT(emission_recon[j])
        transcript_recon.append(transcript)

    cer_gt = CER(transcript_gt, gt_label)
    cer_recon = CER(transcript_recon, gt_label)
    
    # Calculate the L2 norm of the model parameters
    l2_norm = sum(p.pow(2.0).sum() for p in model_g.parameters())

    # Add the L2 regularization term to the loss
    #loss_g += args.lambda_reg * l2_norm


    if trainValid:
        loss_g.backward() 
        optimizer_g.step()
    
    e_loss_g = (loss_g.item(),  loss_ctc.item())
    e_acc_g = (cer_gt.item(), cer_recon.item())

    
    return mel_out, e_loss_g
      


def saveData(args, test_loader, models, epoch, losses):
    
    model_g = models[0].eval()
    vocoder = models[1].eval()
    model_STT = models[2].eval()
    decoder_STT = models[3]

    input, target, target_cl,  data_info = next(iter(test_loader))   
    
    input = input.cuda()
    target = target.cuda()
    voice = torch.squeeze(voice,dim=-1).cuda()
    labels = torch.argmax(target_cl,dim=1)    
    
    with torch.no_grad():
        # run the mdoel
        output = model_g(input)
    
    # mel_out = DTW_align(output, target)
    # output_denorm = data_denorm(mel_out, data_info[0], data_info[1])
    output_denorm = output
    
    
    wav_recon = mel2wav_vocoder(torch.unsqueeze(output_denorm[0],dim=0), vocoder, 1)
    wav_recon = torch.reshape(wav_recon, (len(wav_recon),wav_recon.shape[-1]))
    
    wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)  
    if wav_recon.shape[1] !=  voice.shape[1]:
        p = voice.shape[1] - wav_recon.shape[1]
        p_s = p//2
        p_e = p-p_s
        wav_recon = F.pad(wav_recon, (p_s,p_e))
        
    ##### STT Wav2Vec 2.0
    gt_label = args.word_label[labels[0].item()]
    
    transcript_recon = perform_STT(wav_recon, model_STT, decoder_STT, gt_label, 1)
    
    # save
    wav_recon = np.squeeze(wav_recon.cpu().detach().numpy())
    
    str_tar = args.word_label[labels[0].item()].replace("|", ",")
    str_tar = str_tar.replace(" ", ",")
    
    str_pred = transcript_recon[0].replace("|", ",")
    str_pred = str_pred.replace(" ", ",")
    
    title = "Tar_{}-Pred_{}".format(str_tar, str_pred)
    wavio.write(args.savevoice + '/e{}_{}.wav'.format(str(str(epoch)), title), wav_recon, args.sample_rate_STT, sampwidth=2)


def main(args):
    model_version = 1
    device = torch.device(f'cuda:{args.gpuNum[0]}' if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device: {} '.format(torch.cuda.current_device())) # check
    print('The number of available GPU:{}'.format(torch.cuda.device_count()))
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # define generator
    config_file = os.path.join(args.model_config, 'config_g.json')
    with open(config_file) as f:
        data = f.read()
    if model_version == 0:
        json_config = json.loads(data)
        h_g = AttrDict(json_config)
        model_name = "NeuroTalk"
        model_g = networks.Generator(h_g).cuda()
    elif model_version == 1:
        h_g = mamba_args()
        model_g = networks.Generator_Mamba(h_g).cuda()
        model_name = "Mamba"
    elif model_version == 2:
        h_g = parse_args()
        model_g = networks.Generator_ESN(h_g).cuda()
        model_name = "ESN"
    
    args.sample_rate_mel = args.sampling_rate
    


    if model_version == 2:
        model_g_wout = [model_g.Wout]

    # vocoder HiFiGAN
    # LJ_FT_T2_V3/generator_v3,   
    config_file = os.path.join(os.path.split(args.vocoder_pre)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    vocoder = model_HiFi(h).cuda()
    state_dict_g = torch.load(args.vocoder_pre) #, map_location=args.device)
    vocoder.load_state_dict(state_dict_g['generator'])
    
    # STT Wav2Vec
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    model_STT = bundle.get_model().cuda()
    args.sample_rate_STT = bundle.sample_rate
    decoder_STT = GreedyCTCDecoder(labels=bundle.get_labels())
    args.word_index, args.word_length = word_index(args.word_label, bundle)
 
    # Parallel setting
    model_g = nn.DataParallel(model_g, device_ids=args.gpuNum)
    vocoder = nn.DataParallel(vocoder, device_ids=args.gpuNum)
    model_STT = nn.DataParallel(model_STT, device_ids=args.gpuNum)

    # loss function
    criterion_recon = nn.L1Loss().cuda() #RMSELoss().cuda() 
    criterion_dtw = SoftDTW(gamma=1.0, normalize=True)
    CER = CharErrorRate().cuda()


    # optimizer
    if model_version != 2:
        optimizer_g = torch.optim.AdamW(model_g.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=0.01)
    else:
        optimizer_g = torch.optim.AdamW(model_g_wout, lr=args.lr_g, betas=(0.8, 0.99), weight_decay=0.01)


    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=args.lr_g_decay, last_epoch=-1)

   # create the directory if not exist
    if not os.path.exists(args.logDir):
        os.mkdir(args.logDir)
        
    subDir = os.path.join(args.logDir, args.sub)
    if not os.path.exists(subDir):
        os.mkdir(subDir)        
        
    saveDir = os.path.join(args.logDir, args.sub, args.task)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    logsDir = os.path.join(args.logDir, args.sub, args.task, "logs")
    if not os.path.exists(logsDir):
        os.mkdir(logsDir)
    
    savemodelDir = os.path.join(args.logDir, args.sub, args.task, "savemodel")
    if not os.path.exists(savemodelDir):
        os.mkdir(savemodelDir)
        
    # args.savevoice = saveDir + '/epovoice'
    # if not os.path.exists(args.savevoice):
    #     os.mkdir(args.savevoice)
    args.logs = saveDir + '/logs/' + str(args.batch_size) + '/'
    if not os.path.exists(args.logs):
        os.mkdir(args.logs)
    
    n_num = str(len(os.listdir(args.logs))+1).zfill(4)
    save_model_path = f"{n_num}_model_{model_name}/"

    args.savemodel = saveDir + f'/savemodel/{save_model_path}'
    if not os.path.exists(args.savemodel):
        os.mkdir(args.savemodel)
        

        
    # Load trained model
    start_epoch = 0
    if args.pretrain:
        loc_g = os.path.join(args.trained_model, args.sub, 'BEST_checkpoint_g.pt')

        if os.path.isfile(loc_g):
            print("=> loading checkpoint '{}'".format(loc_g))
            checkpoint_g = torch.load(loc_g, map_location='cpu')
            model_g.load_state_dict(checkpoint_g['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(loc_g))


    if args.resume:
        loc_g = os.path.join(args.savemodel, 'checkpoint_g.pt')

        if os.path.isfile(loc_g):
            print("=> loading checkpoint '{}'".format(loc_g))
            checkpoint_g = torch.load(loc_g, map_location='cpu')
            model_g.load_state_dict(checkpoint_g['state_dict'])
            start_epoch = checkpoint_g['epoch'] + 1
        else:
            print("=> no checkpoint found at '{}'".format(loc_g))

    # Tensorboard setting
    
    log_name = f"{n_num}_model_{model_name}"
    args.writer = SummaryWriter(args.logs+log_name)
    
    # Data loader define
    generator = torch.Generator().manual_seed(args.seed)
    
    if args.task.find("Spoken") == 0:
        args.ta = "sp"
    elif args.task.find("Imagined") == 0:
        args.ta = "im"

    trainset = myDataset(mode=0, data=args.dataLoc+'/'+args.sub, task=args.task, recon=args.ta+"_"+args.recon)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)
    
    valset = myDataset(mode=2, data=args.dataLoc+'/'+args.sub, task=args.task, recon=args.ta+"_"+args.recon)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)

    epoch = start_epoch
    lr_g = 0
    lr_d = 0
    best_loss = 1000
    is_best = False
    epochs_since_improvement = 0
    
    for epoch in range(start_epoch, args.max_epochs):
        
        start_time = time.time()
        
        for param_group in optimizer_g.param_groups:
            lr_g = param_group['lr']

        scheduler_g.step(epoch)

        print("Epoch : %d/%d" %(epoch, args.max_epochs) )
        print("Learning rate for G: %.9f" %lr_g)

        Tr_losses = train(args, train_loader, 
                          (model_g, vocoder, model_STT, decoder_STT), 
                          (criterion_recon, criterion_ctc, CER), 
                          (optimizer_g), 
                          epoch,
                          True) 

        
        Val_losses = train(args, val_loader, 
                           (model_g, vocoder, model_STT, decoder_STT), 
                           (criterion_recon, criterion_ctc, CER), 
                           ([],[]), 
                           epoch,
                           False)
        
        # Save checkpoint
        state_g = {'arch': str(model_g),
                 'state_dict': model_g.state_dict(),
                 'epoch': epoch,
                 'optimizer_state_dict': optimizer_g.state_dict()}
        
        
        # Did validation loss improve?
        loss_total =  Val_losses[3] # default is 0
        is_best = loss_total < best_loss
        best_loss = min(loss_total, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        save_checkpoint(state_g, is_best, args.savemodel, 'checkpoint_g.pt')

        saveData(args, val_loader, (model_g, vocoder, model_STT, decoder_STT), epoch, (Tr_losses, Val_losses))

        time_taken = time.time() - start_time
        print("Time: %.2f\n"%time_taken)
        
    args.writer.flush()

if __name__ == '__main__':

    dataDir = './dataset'
    logDir = './TrainResult'
    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--vocoder_pre', type=str, default='./pretrained_model/UNIVERSAL_V1/g_02500000', help='pretrained vocoder file path')
    parser.add_argument('--trained_model', type=str, default='./pretrained_model', help='trained model for G & D folder path')
    parser.add_argument('--model_config', type=str, default='./models', help='config for G & D folder path')
    parser.add_argument('--dataLoc', type=str, default=dataDir)
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--logDir', type=str, default=logDir)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--prefreeze', type=bool, default=False)
    parser.add_argument('--gpuNum', type=list, default=[0])
    parser.add_argument('--batch_size', type=int, default=26)
    parser.add_argument('--sub', type=str, default='sub1')
    parser.add_argument('--task', type=str, default='SpokenEEG')
    parser.add_argument('--recon', type=str, default='Y_mel')

    
    args = parser.parse_args()
    
    with open(args.config) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    main(args)        
    
    
    
