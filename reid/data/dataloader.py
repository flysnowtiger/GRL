from __future__ import print_function
import os.path as osp
from torch.utils.data import DataLoader
from reid.dataset import get_sequence
from reid.data import seqtransforms as T
from reid.data import SeqTrainPreprocessor
from reid.data import SeqTestPreprocessor
from reid.data import RandomPairSampler, RandomPairSamplerForMars
from reid.data.video_loader import VideoDataset


def get_data(dataset_name, split_id, data_dir, batch_size, seq_len, seq_srd, workers, only_eval):

    if dataset_name != 'mars' and dataset_name != 'duke':
        root = osp.join(data_dir, dataset_name)
        dataset = get_sequence(dataset_name, root, split_id=split_id,
                               seq_len=seq_len, seq_srd=seq_srd, num_val=1, download=True)
        train_set = dataset.trainval
        num_classes = dataset.num_trainval_ids
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_processor = SeqTrainPreprocessor(train_set, dataset, seq_len,
                                               transform=T.Compose([T.RectScale(256, 128),
                                                                    T.RandomHorizontalFlip(),
                                                                    T.RandomSizedEarser(),
                                                                    T.ToTensor(), normalizer]))

        query_processor = SeqTestPreprocessor(dataset.query, dataset, seq_len,
                                              transform=T.Compose([T.RectScale(256, 128),
                                                                   T.ToTensor(), normalizer]))

        gallery_processor = SeqTestPreprocessor(dataset.gallery, dataset, seq_len,
                                                transform=T.Compose([T.RectScale(256, 128),
                                                                     T.ToTensor(), normalizer]))

        train_loader = DataLoader(train_processor, batch_size=batch_size, num_workers=workers,
                                  sampler=RandomPairSampler(train_set), pin_memory=True, drop_last=True)

        query_loader = DataLoader(query_processor, batch_size=8,
                                  num_workers=workers, shuffle=False, pin_memory=True, drop_last=False)

        gallery_loader = DataLoader(gallery_processor, batch_size=8,
                                    num_workers=workers, shuffle=False, pin_memory=True, drop_last=False)

    else:
        dataset = get_sequence(dataset_name)  # mars数据集
        train_set = dataset.train  # 8298

        num_classes = dataset.num_train_pids  # 625

        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_processor = VideoDataset(train_set, seq_len=seq_len, sample='rrs_train',
                                       transform=T.Compose([T.RectScale(256, 128),
                                                            T.RandomHorizontalFlip(),
                                                            T.RandomSizedEarser(),
                                                            T.ToTensor(), normalizer]))


        if only_eval:
            sampler_method = 'dense'
            batch_size_eval = 1
        else:
            sampler_method = 'rrs_test'
            batch_size_eval = 30
        query_processor = VideoDataset(dataset.query, seq_len=seq_len, sample=sampler_method,
                                       transform=T.Compose([T.RectScale(256, 128),
                                                            T.ToTensor(), normalizer]))

        gallery_processor = VideoDataset(dataset.gallery, seq_len=seq_len, sample=sampler_method,
                                         transform=T.Compose([T.RectScale(256, 128),
                                                              T.ToTensor(), normalizer]))

        train_loader = DataLoader(train_processor, batch_size=batch_size, num_workers=workers,
                                  sampler=RandomPairSamplerForMars(train_set), pin_memory=True, drop_last=True)

        query_loader = DataLoader(query_processor, batch_size=batch_size_eval, shuffle=False, pin_memory=True, drop_last=False)

        gallery_loader = DataLoader(gallery_processor, batch_size=batch_size_eval, shuffle=False, pin_memory=True, drop_last=False)

    return dataset, num_classes, train_loader, query_loader, gallery_loader
