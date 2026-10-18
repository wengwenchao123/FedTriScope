"""Train FedTriScope or evaluate a trusted local checkpoint."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from fedtriscope.model import FedTriScope, SharedBackbone
from fedtriscope.data import CityData
from fedtriscope.client import Client, Server
from fedtriscope.trainer import FedTSBTrainer

ROOT = Path(__file__).resolve().parent
CITIES = ['metr-la', 'pems-bay', 'shenzhen', 'chengdu']
DEFAULTS = dict(target_city='metr-la', data_list=CITIES, dim=8, lag=12, horizon=6,
                target_days=3, target_steps_per_day=288, normalization='full',
                batch_size=32, num_rounds=90, local_epochs=150, target_epochs=50,
                lr_init=0.001, seed=0, tsb_target_lr=0.0002, tsb_source_lr=0.000001,
                tsb_projection='tsb')

def init_seed(seed, device):
    random.seed(1234 + seed)
    np.random.seed(1 + seed)
    torch.manual_seed(12 + seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(123 + seed)
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def read_args(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    preliminary.add_argument('--config', type=Path)
    known, _ = preliminary.parse_known_args(argv)
    defaults = DEFAULTS.copy()
    if known.config:
        loaded = json.loads(known.config.read_text(encoding='utf-8'))
        unknown = set(loaded)-set(defaults)
        if unknown:
            raise ValueError(f'Unknown config fields: {sorted(unknown)}')
        defaults.update(loaded)
    p = argparse.ArgumentParser(parents=[preliminary], allow_abbrev=False)
    p.add_argument('--mode', choices=['train', 'test'], default='train')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--data_path', type=Path, default=ROOT/'data')
    p.add_argument('--log_dir', type=Path)
    p.add_argument('--checkpoint', type=Path)
    for key, value in defaults.items():
        if key == 'data_list':
            p.add_argument('--data_list', nargs='+', default=value, choices=CITIES)
        else:
            p.add_argument('--'+key, type=type(value), default=value)
    a = p.parse_args(argv)
    if a.mode == 'test' and not a.checkpoint:
        p.error('--mode test requires --checkpoint (load only trusted checkpoints)')
    if a.target_city not in CITIES or a.target_city not in a.data_list:
        p.error('target_city must be one of the participating four cities')
    if len(a.data_list) != len(set(a.data_list)) or len(a.data_list)<2:
        p.error('Use at least two distinct cities')
    for key in ['dim','lag','horizon','target_days','batch_size','num_rounds','local_epochs']:
        if getattr(a,key)<=0: p.error(key+' must be positive')
    if min(a.target_epochs,a.target_steps_per_day,a.seed)<0:
        p.error('Epochs, steps per day and seed must be nonnegative')
    for key in ['lr_init','tsb_target_lr','tsb_source_lr']:
        value=getattr(a,key)
        if not np.isfinite(value) or value<0 or (key!='tsb_source_lr' and value==0):
            p.error(key+' is invalid')
    if a.normalization not in ['full','train'] or a.tsb_projection not in ['tsb','none']:
        p.error('Invalid normalization or projection')
    a.data_path = str(a.data_path.resolve())
    a.config = str(a.config) if a.config else None
    a.checkpoint = str(a.checkpoint) if a.checkpoint else None
    a.log_dir = str(a.log_dir.resolve()) if a.log_dir else None
    if a.device.startswith('cuda'):
        if not torch.cuda.is_available():
            p.error('CUDA unavailable; pass --device cpu explicitly')
        torch.cuda.set_device(torch.device(a.device))
    return a


def main(argv=None):
    args = read_args(argv)
    checkpoint = None
    if args.mode == 'test':
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        saved = checkpoint['args'].copy()
        for key in ['mode','device','data_path','log_dir','checkpoint']:
            saved[key] = getattr(args,key)
        args = argparse.Namespace(**saved)
    if args.log_dir is None:
        args.log_dir = str(ROOT/'runs'/args.target_city/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output=Path(args.log_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use an empty output directory: {output}')
    init_seed(args.seed,args.device)
    clients=[]
    for city in ([args.target_city] if checkpoint else args.data_list):
        data=CityData(city,args)
        model=FedTriScope(data.num_nodes,data.steps_per_day,args.lag,args.horizon,args.dim)
        clients.append(Client(model,data,args))
    backbone=SharedBackbone(args.lag,args.horizon,args.dim).to(args.device)
    trainer=FedTSBTrainer(Server(backbone),clients,args)
    (output/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    if checkpoint:
        target=trainer.target
        target.model.load_state_dict(checkpoint['clients'][args.target_city],strict=True)
        scaler=checkpoint['scalers'][args.target_city]
        target.data.mean,target.data.std=scaler['mean'],scaler['std']
        for dataset in (target.data.train,target.data.test):
            dataset.mean,dataset.std=scaler['mean'],scaler['std']
        if checkpoint['time_sources'][args.target_city]!=target.data.time_source:
            raise ValueError('Checkpoint calendar protocol differs from the current data')
        loader=DataLoader(target.data.test,batch_size=args.batch_size,shuffle=False)
        result=trainer._evaluate(loader)
        trainer.save_result(result)
        return result
    return trainer.train_fl()

if __name__ == '__main__':
    main()
