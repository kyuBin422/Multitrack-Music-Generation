from functools import reduce
import json
import numpy as np
import pandas as pd
import requests
from musixmatch import Musixmatch
import muspy
import pickle

CodeInstrumentMap = {
    # Pianos
    0: "piano",
    1: "piano",
    2: "piano",
    3: "piano",
    4: "electric-piano",
    5: "electric-piano",
    6: "harpsichord",
    7: "clavinet",
    # Chromatic Percussion
    8: "celesta",
    9: "glockenspiel",
    10: "music-box",
    11: "vibraphone",
    12: "marimba",
    13: "xylophone",
    14: "tubular-bells",
    15: "dulcimer",
    # Organs
    16: "organ",
    17: "organ",
    18: "organ",
    19: "church-organ",
    20: "organ",
    21: "accordion",
    22: "harmonica",
    23: "bandoneon",
    # Guitars
    24: "nylon-string-guitar",
    25: "steel-string-guitar",
    26: "electric-guitar",
    27: "electric-guitar",
    28: "electric-guitar",
    29: "electric-guitar",
    30: "electric-guitar",
    31: "electric-guitar",
    # Basses
    32: "bass",
    33: "electric-bass",
    34: "electric-bass",
    35: "electric-bass",
    36: "slap-bass",
    37: "slap-bass",
    38: "synth-bass",
    39: "synth-bass",
    # Strings
    40: "violin",
    41: "viola",
    42: "cello",
    43: "contrabass",
    44: "strings",
    45: "strings",
    46: "harp",
    47: "timpani",
    # Ensemble
    48: "strings",
    49: "strings",
    50: "synth-strings",
    51: "synth-strings",
    52: "voices",
    53: "voices",
    54: "voices",
    55: "orchestra-hit",
    # Brass
    56: "trumpet",
    57: "trombone",
    58: "tuba",
    59: "trumpet",
    60: "horn",
    61: "brasses",
    62: "synth-brasses",
    63: "synth-brasses",
    # Reed
    64: "soprano-saxophone",
    65: "alto-saxophone",
    66: "tenor-saxophone",
    67: "baritone-saxophone",
    68: "oboe",
    69: "english-horn",
    70: "bassoon",
    71: "clarinet",
    # Pipe
    72: "piccolo",
    73: "flute",
    74: "recorder",
    75: "pan-flute",
    76: None,
    77: None,
    78: None,
    79: "ocarina",
    # Synth Lead
    80: "lead",
    81: "lead",
    82: "lead",
    83: "lead",
    84: "lead",
    85: "lead",
    86: "lead",
    87: "lead",
    # Synth Pad
    88: "pad",
    89: "pad",
    90: "pad",
    91: "pad",
    92: "pad",
    93: "pad",
    94: "pad",
    95: "pad",
    # Synth Effects
    96: None,
    97: None,
    98: None,
    99: None,
    100: None,
    101: None,
    102: None,
    103: None,
    # Ethnic
    104: "sitar",
    105: "banjo",
    106: "shamisen",
    107: "koto",
    108: "kalimba",
    109: "bag-pipe",
    110: "violin",
    111: "shehnai",
    # Percussive
    112: None,
    113: None,
    114: None,
    115: None,
    116: None,
    117: "melodic-tom",
    118: "synth-drums",
    119: "synth-drums",
    # Sound effects
    120: None,
    121: None,
    122: None,
    123: None,
    124: None,
    125: None,
    126: None,
    127: None,
}

with open('md5_to_paths.json') as f:
    md5_to_paths = json.load(f)

with open('match_scores.json') as f:
    match_scores = json.load(f)

lahk_mds_match = {}
for mds in match_scores.keys():
    for lahk in match_scores[mds].keys():
        if lahk not in lahk_mds_match:
            lahk_mds_match[lahk] = [mds, match_scores[mds][lahk]]
        elif match_scores[mds][lahk] >= lahk_mds_match[lahk][1]:
            lahk_mds_match[lahk] = [mds, match_scores[mds][lahk]]

mxm_779k_matches = {}
with open('mxm_779k_matches.txt', encoding="utf8") as f:
    for row in f.readlines():
        mxm_779k_matches[row.split('<SEP>')[-3]] = row.split('<SEP>')[0]

musixmatch = Musixmatch('66bee8626e56107c18fbecf14e8afed1')

md5_to_lyrics = {}
for track_id, mxm_id in mxm_779k_matches.items():
    try:
        md5 = max(match_scores[mxm_id])
        # '13819538'
        lyrics = musixmatch.track_lyrics_get(track_id)['message']['body']['lyrics']['lyrics_body'].split('*')[0]
        md5_to_lyrics[md5]['lyrics'] = lyrics
        music = muspy.load_json('data/lmd_full/processed/json/' + md5[0] + '/' + md5 + '.mid.json')
        Bar = music.get_real_end_time() * music.tempos[0].qpm / music.time_signatures[0].numerator
        Bar = round(Bar)
        TimeSignature = str(music.time_signatures[0].numerator) + '/' + str(
            music.time_signatures[0].denominator)
        Tempo = 'Slow' if music.tempos[0].qpm <= 76 else 'Moderato' if music.tempos[0].qpm < 120 else 'Fast'
        Key = '' if len(music.key_signatures) == 0 else music.key_signatures[0].mode
        Instrument = [CodeInstrumentMap[track.program] for track in music.tracks]

        md5_to_lyrics[md5]['attribute'] = {
            'Bar': Bar,
            'TimeSignature': TimeSignature,
            'Tempo': Tempo,
            'Key': Key,
            'Instrument': Instrument,
        }
        with open('md5_to_lyrics.pickle', 'wb') as f:
            pickle.dump(md5_to_lyrics, f)
    except Exception as e:
        print(e)
