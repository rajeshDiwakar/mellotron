import os
import time
import argparse
import math
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import load_model
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams
import glob

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

if not os.path.isfile('mycreds.txt'):
    with open('mycreds.txt','w') as f:
        f.write('{"access_token": "ya29.a0AfH6SMC_aOt4BLq-OQ1oN4txyT5Guk9KMeEzqYJDjo4AkqD0fMJnIdQm4TGz3PQit8qNa-QEg3hdg66ic2pLErifxwsEhgPP-MIa947Ayigh8c5czN64T9IxCyLkR2M-5ygdjOhV5OzuXw-O6LfBJG9vBwMkyg9OKL0", "client_id": "883051571054-2e0bv2mjqra6i3cd6c915hkjgtdutct0.apps.googleusercontent.com", "client_secret": "NmzemQWSeUm_WWTbmUJi5xt7", "refresh_token": "1//0gE7zkyCPJ4RpCgYIARAAGBASNwF-L9IrISJx8AG8doLKF1C8RMbuvkqS6BsxGXaYJfqlB-RbrtmIESmVIA2krp-rK-Ylm26klmU", "token_expiry": "2020-07-29T16:47:41Z", "token_uri": "https://oauth2.googleapis.com/token", "user_agent": null, "revoke_uri": "https://oauth2.googleapis.com/revoke", "id_token": null, "id_token_jwt": null, "token_response": {"access_token": "ya29.a0AfH6SMC_aOt4BLq-OQ1oN4txyT5Guk9KMeEzqYJDjo4AkqD0fMJnIdQm4TGz3PQit8qNa-QEg3hdg66ic2pLErifxwsEhgPP-MIa947Ayigh8c5czN64T9IxCyLkR2M-5ygdjOhV5OzuXw-O6LfBJG9vBwMkyg9OKL0", "expires_in": 3599, "refresh_token": "1//0gE7zkyCPJ4RpCgYIARAAGBASNwF-L9IrISJx8AG8doLKF1C8RMbuvkqS6BsxGXaYJfqlB-RbrtmIESmVIA2krp-rK-Ylm26klmU", "scope": "https://www.googleapis.com/auth/drive", "token_type": "Bearer"}, "scopes": ["https://www.googleapis.com/auth/drive"], "token_info_uri": "https://oauth2.googleapis.com/tokeninfo", "invalid": false, "_class": "OAuth2Credentials", "_module": "oauth2client.client"}')

        # {"access_token": "ya29.a0AfH6SMCDGn8XAOVlzeT47aIMf7QlauIfWz3G9fXrRTyX0JgSllcpHrAIuj6s6zqNTI0kK46c4LmVQp2svHpCSltdQrSgLo-74UtFWv4mdUX0Rnt5TxM7I_OaewjmLl6vH8wmrk1bccDAWBY_-vTeBI-eEedfSNRQu4Mc", "client_id": "883051571054-2e0bv2mjqra6i3cd6c915hkjgtdutct0.apps.googleusercontent.com", "client_secret": "NmzemQWSeUm_WWTbmUJi5xt7", "refresh_token": "1//0gE7zkyCPJ4RpCgYIARAAGBASNwF-L9IrISJx8AG8doLKF1C8RMbuvkqS6BsxGXaYJfqlB-RbrtmIESmVIA2krp-rK-Ylm26klmU", "token_expiry": "2020-08-09T09:46:00Z", "token_uri": "https://oauth2.googleapis.com/token", "user_agent": null, "revoke_uri": "https://oauth2.googleapis.com/revoke", "id_token": null, "id_token_jwt": null, "token_response": {"access_token": "ya29.a0AfH6SMCDGn8XAOVlzeT47aIMf7QlauIfWz3G9fXrRTyX0JgSllcpHrAIuj6s6zqNTI0kK46c4LmVQp2svHpCSltdQrSgLo-74UtFWv4mdUX0Rnt5TxM7I_OaewjmLl6vH8wmrk1bccDAWBY_-vTeBI-eEedfSNRQu4Mc", "expires_in": 3599, "scope": "https://www.googleapis.com/auth/drive", "token_type": "Bearer"}, "scopes": ["https://www.googleapis.com/auth/drive"], "token_info_uri": "https://oauth2.googleapis.com/tokeninfo", "invalid": false, "_class": "OAuth2Credentials", "_module": "oauth2client.client"}


gauth = GoogleAuth()
# Try to load saved client credentials
gauth.LoadCredentialsFile("mycreds.txt")
# if gauth.credentials is None:
#     # Authenticate if they're not there
#     gauth.LocalWebserverAuth()
if gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
# Save the current credentials to a file
gauth.SaveCredentialsFile("mycreds.txt")

# drive = GoogleDrive(gauth)

def authorize_drive():
    # global drive
    global gauth
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    # if gauth.credentials is None:
    #     # Authenticate if they're not there
    #     gauth.LocalWebserverAuth()
    if gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")

    drive = GoogleDrive(gauth)

    return drive


# def validate_parent_id(parent_id):
#     global drive
#     file_list = drive.ListFile({'q': f"title='{folder_name}' and trashed=false and mimeType='application/vnd.google-apps.folder'"}).GetList()
#         if len(file_list) > 1:
#             raise ValueError('There are multiple folders with that specified folder name')
#         elif len(file_list) == 0:
#             raise ValueError('No folders match that specified folder name')


def upload_to_drive(list_files,parent_id):
    # global drive
    drive = authorize_drive()
    # parent_id = ''# parent id
    drive_files = drive.ListFile({'q': "'%s' in parents and trashed=false"%parent_id}).GetList()
    drive_files = {f['title']:f for f in drive_files}
    for path in list_files:
        if not os.path.isfile(path): continue
        d,f = os.path.split(path)
        # check if file already exists and trash it
        if f in drive_files:
                drive_files[f].Trash()

        file = drive.CreateFile({'title': f, 'parents': [{'id': parent_id}]})
        file.SetContentFile(path)
        file.Upload()

def download_checkpoints(parent_id,checkpoint,root_dir='outdir'):
    drive = authorize_drive()
    downloaded_files = []
    os.makedirs(root_dir,exist_ok=True)
    # checkpoint = ''
    # file_list = drive.ListFile({'q': "title contains 'My Awesome File' and trashed=false"}).GetList()
    ckpt_path = os.path.join(root_dir,checkpoint)
    file_list = drive.ListFile({'q': "'%s' in parents and trashed=false"%parent_id}).GetList()  #check if it is iterator
    # print(file_list)
    for f in file_list:
        if f['title'].lower() == checkpoint:
            file_id = f['id']
            file = drive.CreateFile({'id': file_id})
            file.GetContentFile(ckpt_path)
            downloaded_files.append(ckpt_path)
#

    if os.path.isfile(ckpt_path):
        # with open(ckpt_path) as f:
        #     ckpt_data = f.read().split('\n')
        # if len(ckpt_data):
        #     ckpt_data = ckpt_data[0].split(':')[-1].strip().strip('" ')
        #     weight_name = os.path.basename(ckpt_data)
        #     for f in file_list:
        #         if f['title'].startswith(weight_name):
        #             file_id = f['id']
        #             file = drive.CreateFile({'id': file_id})
        #             file.GetContentFile(os.path.join(root_dir,f['title']))
        #             downloaded_files.append(os.path.join(root_dir,f['title']))
        pass
    else:
        log('checkpoint file not found in drive')

    print('Downloaded following files\n%s'%'\n'.join(downloaded_files))

    return ckpt_path




def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams,
                           speaker_ids=trainset.speaker_ids)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn, train_sampler


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):

    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, reduced_val_loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams,args):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    tstart = time.time()
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn, train_sampler = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            if checkpoint_path.startswith('pid'):
                checkpoint = os.path.basename(checkpoint_path)
                checkpoint_path = download_checkpoints(args.pid,checkpoint,output_directory)
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    unsaved_data=False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        if args.max_duration and time.time()-tstart>args.max_duration:
            if unsaved_data:
                checkpoint_path = os.path.join(
                    output_directory, "checkpoint_{}".format(iteration))
                save_checkpoint(model, optimizer, learning_rate, iteration,
                                checkpoint_path)
                unsaved_data = False
                if args.pid:
                    try:
                        log_files = glob.glob(os.path.join(output_directory,log_directory,'*'))
                        upload_to_drive([checkpoint_path]+log_files,args.pid)
                    except Exception as e:
                        print('error while uploading to drive\n%s'%str(e))
            break
        print("Epoch: {}".format(epoch))
        unsaved_data = True
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            if iteration > 0 and iteration % hparams.learning_rate_anneal == 0:
                learning_rate = max(
                    hparams.learning_rate_min, learning_rate * 0.5)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()

            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, criterion, valset, iteration,
                        hparams.batch_size, n_gpus, collate_fn, logger,
                        hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)
                    unsaved_data = False
                    if args.pid:
                        try:
                            log_files = glob.glob(os.path.join(output_directory,log_directory,'*'))
                            upload_to_drive([checkpoint_path]+log_files,args.pid)
                        except Exception as e:
                            print('error while uploading to drive\n%s'%str(e))

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--pid', type=str,
                        required=False, help='drive folder parent id')
    parser.add_argument('--max_duration',default=0,type=int,help='max duration for application in seconds')
    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams,args)
