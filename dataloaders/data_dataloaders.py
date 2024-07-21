import os
import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_lpr4m_retrieval import LPR4M_DataLoader, LPR4M_TrainDataLoader
from dataloaders.dataloader_mf_retrieval import MF_TrainDataLoader

        
def dataloader_lpr4m_train(args, tokenizer):
    lpr4m_dataset = LPR4M_TrainDataLoader(
        data_path=os.path.join(args.features_path, "_training_data.txt"),
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(lpr4m_dataset)
    dataloader = DataLoader(
        lpr4m_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(lpr4m_dataset), train_sampler

def dataloader_lpr4m_test(args, tokenizer, subset="test"):
    lpr4m_testset = LPR4M_DataLoader(
        data_path=os.path.join(args.features_path, "_test_data.txt"),
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader = DataLoader(
        lpr4m_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(lpr4m_testset)

def dataloader_mf_train(args, tokenizer):
    mf_dataset = MF_TrainDataLoader(
        data_path=os.path.join(args.features_path, "movingfashion_train_data.txt"),
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(mf_dataset)
    dataloader = DataLoader(
        mf_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(mf_dataset), train_sampler



DATALOADER_DICT = {}
DATALOADER_DICT["lpr4m"] = {"train":dataloader_lpr4m_train, "val":dataloader_lpr4m_test, "test":None}
DATALOADER_DICT["mf"] = {"train":dataloader_mf_train, "val":dataloader_lpr4m_test, "test":None}
