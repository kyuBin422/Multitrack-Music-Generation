"""Data loader."""
import argparse
import logging
import pathlib
import pickle
import pprint
import sys
from functools import reduce

import numpy as np
import torch
import torch.utils.data
import tqdm

import representation
import utils
from tokenizer import Tokenizer


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        choices=("transformer", "retnet", "mamba"),
        required=True,
        help="dataset key",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd", "lmd_full", "snd"),
        required=True,
        help="dataset key",
    )
    parser.add_argument("-n", "--names", type=pathlib.Path, help="input names")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    # Data
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=8,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    parser.add_argument(
        "--aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use data augmentation",
    )
    parser.add_argument(
        "--max_seq_len",
        default=1024,
        type=int,
        help="maximum sequence length",
    )
    parser.add_argument(
        "--max_beat",
        default=256,
        type=int,
        help="maximum number of beats",
    )
    # Others
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        help="number of jobs (deafult to `min(batch_size, 8)`)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    parser.add_argument("--tokenizer_path", default='tokenizer.model', help="the path of tokenizer model")
    parser.add_argument("--comment_path", default='md5_to_attribute.pickle')
    return parser.parse_args(args=args, namespace=namespace)


def pad(data, max_len=None):
    if max_len is None:
        max_len = max(len(x) for x in data)
    else:
        for x in data:
            assert len(x) <= max_len
    if data[0].ndim == 1:
        padded = [np.pad(x, (0, max_len - len(x))) for x in data]
    elif data[0].ndim == 2:
        padded = [np.pad(x, ((0, max_len - len(x)), (0, 0))) for x in data]
    else:
        raise ValueError("Got 3D data.")
    return np.stack(padded)


def get_mask(data):
    max_seq_len = max(len(sample) for sample in data)
    mask = torch.zeros((len(data), max_seq_len), dtype=torch.bool)
    for i, seq in enumerate(data):
        mask[i, : len(seq)] = 1
    return mask


class MusicDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            filename,
            data_dir,
            encoding,
            tokenizer,
            lmd_full_comment,
            max_seq_len=None,
            max_beat=None,
            use_csv=False,
            use_augmentation=False,
    ):
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        with open(filename) as f:
            self.names = [line.strip() for line in f if line]
        self.encoding = encoding
        self.max_seq_len = max_seq_len
        self.max_beat = max_beat
        self.use_csv = use_csv
        self.use_augmentation = use_augmentation
        self.tokenizer = tokenizer
        self.lmd_full_comment = lmd_full_comment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # Get the name
        name = self.names[idx]

        # Load data
        if self.use_csv:
            notes = utils.load_csv(self.data_dir / f"{name}.csv")
        else:
            try:
                notes = np.load(self.data_dir / f"{name}.npy")
            except:
                notes = np.load(self.data_dir / f"{name[:-4]}.npy")

        # Check the shape of the loaded notes
        assert notes.shape[1] == 9

        # Data augmentation
        if self.use_augmentation:
            # Shift all the pitches for k semitones (k~Uniform(-5, 6))
            pitch_shift = np.random.randint(-5, 7)
            notes[:, 2] = np.clip(notes[:, 2] + pitch_shift, 0, 127)

            # Randomly select a starting beat
            n_beats = notes[-1, 0] + 1
            if n_beats > self.max_beat:
                trial = 0
                # Avoid section with too few notes
                while trial < 10:
                    start_beat = np.random.randint(n_beats - self.max_beat)
                    end_beat = start_beat + self.max_beat
                    sliced_notes = notes[
                        (notes[:, 0] >= start_beat) & (notes[:, 0] < end_beat)
                        ]
                    if len(sliced_notes) > 10:
                        break
                    trial += 1
                sliced_notes[:, 0] = sliced_notes[:, 0] - start_beat
                notes = sliced_notes

        # Trim sequence to max_beat
        elif self.max_beat is not None:
            n_beats = notes[-1, 0] + 1
            if n_beats > self.max_beat:
                notes = notes[notes[:, 0] < self.max_beat]

        # Encode the notes
        seq = representation.encode_notes(notes, self.encoding)
        np.set_printoptions(threshold=sys.maxsize)
        # Trim sequence to max_seq_len
        if self.max_seq_len is not None and len(seq) > self.max_seq_len:
            seq = np.concatenate((seq[: self.max_seq_len - 1], seq[-1:]))

        # tokenizer the prompt to tensor
        if name[2:] in list(self.lmd_full_comment.keys()):
            prompt = reduce(lambda x, y: x if len(x) > len(y) else y, self.lmd_full_comment[name[2:]]['comment'])
            prompt = np.array(self.tokenizer.encode(prompt, bos=True, eos=False))
            attribute = np.array(
                self.tokenizer.encode(str(self.lmd_full_comment[name[2:]]['attribute']), bos=True, eos=False))
        else:
            prompt = np.array(self.tokenizer.encode("", bos=True, eos=False))
            attribute = np.array(self.tokenizer.encode("", bos=True, eos=False))
        return {"name": name, "seq": seq, "prompt": prompt, "attribute": attribute}

    @classmethod
    def collate(cls, data):
        seq = [sample["seq"] for sample in data]
        max_len = max(len(x) for x in seq)
        prompt = [sample["prompt"][:max_len] for sample in data]
        attribute = [sample["attribute"] for sample in data]

        return {
            "name": [sample["name"] for sample in data],
            "seq": torch.tensor(pad(seq, max_len=max_len), dtype=torch.long),
            "prompt": torch.tensor(pad(prompt, max_len=max_len), dtype=torch.long),
            "attribute": torch.tensor(pad(attribute, max_len=max_len), dtype=torch.long),
            "seq_len": torch.tensor([len(s) for s in seq], dtype=torch.long),
            "mask": get_mask(seq),
        }
