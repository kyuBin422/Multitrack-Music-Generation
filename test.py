import pickle

data=pickle.load(open('md5_to_attribute.pickle','rb'))

#


# Copyright (c) Meta Platforms, Inc. and affiliates.
# # This software may be used and distributed according to the terms of the GNU General Public License version 3.
#
# import os
# from logging import getLogger
# from typing import List
#
# from sentencepiece import SentencePieceProcessor
#
# logger = getLogger()
#
#
# class Tokenizer:
#     def __init__(self, model_path: str):
#         # reload tokenizer
#         assert os.path.isfile(model_path), model_path
#         self.sp_model = SentencePieceProcessor(model_file=model_path)
#         logger.info(f"Reloaded SentencePiece model from {model_path}")
#
#         # BOS / EOS token IDs
#         self.n_words: int = self.sp_model.vocab_size()
#         self.bos_id: int = self.sp_model.bos_id()
#         self.eos_id: int = self.sp_model.eos_id()
#         self.pad_id: int = self.sp_model.pad_id()
#         logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
#         assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
#
#     def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
#         assert type(s) is str
#         t = self.sp_model.encode(s)
#         if bos:
#             t = [self.bos_id] + t
#         if eos:
#             t = t + [self.eos_id]
#         return t
#
#     def decode(self, t: List[int]) -> str:
#         return self.sp_model.decode(t)
#
#
# if __name__ == '__main__':
#     T = Tokenizer('tokenizer.model')
#
#     print(T.encode("X:1"
#                    "T:Generated Melody in C Major"
#                    "M:4/4"
#                    "L:1/4"
#                    "K:C"
#                    "E2 B1 D1 C2 A1 C1 C2 G2 B4 A4 F1 D2 E4 C2 E2 E4"
#                    "W:Sun-ny days, clear blue skies"
#                    "W:Joy-ful hearts, as time flies"
#                    "W:Laugh-ter fills the air we breathe"
#                    "W:Pre-cious mo-ments, we be-lieve", bos=False, eos=False))
#     print(T.decode([306, 29915, 345]))
#
#     from music21 import stream, note, metadata
#
#     # Create a new music stream
#     score = stream.Score()
#     part = stream.Part()
#     score.append(part)
#
#     # Add metadata to the score
#     score.metadata = metadata.Metadata()
#     score.metadata.title = "C Major Composition"
#     score.metadata.composer = "AI Composer"
#
#     # Define a simple melody in C major
#     notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
#              'C5', 'B4', 'A4', 'G4', 'F4', 'E4', 'D4', 'C4']
#
#     # Create and add notes to the part
#     for pitch in notes:
#         n = note.Note(pitch)
#         n.duration.quarterLength = 1  # setting duration to 1 quarter note
#         part.append(n)
#
#     # Export the music to MusicXML
#     # musicxml_path = "/mnt/data/C_Major_Composition.musicxml"
#     # score.write('musicxml', fp=musicxml_p
#     # ath)
#     # Export the music to MusicXML
#     musicxml_path = "C_Major_Composition.musicxml"
#     score.write('musicxml', fp=musicxml_path)
#
#
#


