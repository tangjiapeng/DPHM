import sys
import numpy as np
import torch
import os
import os.path as osp
from tqdm import tqdm
from collections import OrderedDict

class MediaPipeLandmakrs:
    def __init__(self):
        self.mouth = {
        'outer': {
            'l_middle': 61,
            'l_half_upper': 40,
            'l_peak': 37,
            'peak':0,
            'r_peak': 267,
            'r_half_upper': 270,
            'r_middle': 291,
            'r_half_lower': 321,
            'r_bottom': 314,
            'l_bottom': 84,
            'l_half_lower': 91
        },
        'inner': {
            'l_middle': 78,
            'l_half_upper': 80,
            'l_peak': 82,
            'peak':13,
            'r_peak': 312,
            'r_half_upper': 310,
            'r_middle': 308,
            'r_half_lower': 318,
            'r_bottom': 317,
            'l_bottom': 87,
            'l_half_lower': 88
        }
        }

        self.nose = {
            'tip': 4,
            'bottom': 94,
            #'l_bottom': 64,
            #'r_bottom': 294,
            'top': 6
        }

        self.eyes = {

        'left': {
            'r_tip': 133,
            'c_bottom_l': 144,
            'c_bottom_r': 153,
            'l_tip': 33,
            'c_top_l': 160,
            'c_top_r': 157,
            #'pupil': 468,

        }
        ,
        'right': {
            'l_tip': 382,
            'c_bottom_l': 380,
            'c_bottom_r': 373,
            'r_tip': 263,
            'c_top_r': 387,
            'c_top_l': 384
            #'pupil': 474
        }
        }

        self.jawline = {
            'left': {
                'eye_height': 34, #127,
                'nose_height': 177, #132,
                'mouth_height': 138, #172,
                'bottom_left': 169#150
            },
            'bottom': 175, #152,
            'rigth': {
                'eye_height': 264, #356,
                'nose_height': 401, #361,
                'mouth_height': 367, #397,
                'bottom_right': 394, #379
            }
        }

        self.irises = {'l_iris': 468,
                     'r_iris': 473}

        self.face = {'mouth': self.mouth, 'nose': self.nose, 'eyes': self.eyes, 'jawline': self.jawline, 'irises': self.irises}

    def rec_len(self, d):
        n = 0
        for k in d.keys():
            if isinstance(d[k], dict):
                n += self.rec_len(d[k])
            else:
                n += 1
        return n

    def __len__(self):
        return self.rec_len(self.face)

    def get_lms(self):
        ret = []
        for key in self.face:
            part = self.face[key]
            for k in part.keys():
                if isinstance(part[k], dict):
                   for kk in part[k].keys():
                       ret.append(part[k][kk])
                else:
                    ret.append(part[k])
        return np.array(ret)
lm_inds_mp = MediaPipeLandmakrs().get_lms()


#####  pipnet --to -- landmarks 68
DLIB_68_TO_WFLW_98_IDX_MAPPING = OrderedDict()
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(0,17),range(0,34,2)))) # jaw | 17 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(17,22),range(33,38)))) # left upper eyebrow points | 5 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(22,27),range(42,47)))) # right upper eyebrow points | 5 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(27,36),range(51,60)))) # nose points | 9 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({36:60}) # left eye points | 6 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({37:61})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({38:63})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({39:64})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({40:65})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({41:67})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({42:68}) # right eye | 6 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({43:69})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({44:71})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({45:72})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({46:73})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({47:75})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(48,68),range(76,96)))) # mouth points | 20 pts

IBU68_index_into_WFLW = []
for i in range(68):
    IBU68_index_into_WFLW.append(DLIB_68_TO_WFLW_98_IDX_MAPPING[i])
IBU68_index_into_WFLW = np.array(IBU68_index_into_WFLW)


#### FLMAE vertices -- to -- landmarks 68
FLAME_LANDMARK_INDICES = np.array([2212, 3060, 3485, 3384, 3386, 3389, 3418, 3395, 3414, 3598, 3637, 3587, 3582, 3580, 3756, 2012, 730, 1984,
         3157, 335, 3705, 3684, 3851, 3863, 16, 2138, 571, 3553, 3561, 3501, 3526, 2748, 2792, 3556, 1675, 1612, 2437, 2383, 2494, 3632, 2278,
         2296, 3833, 1343, 1034, 1175, 884, 829, 2715, 2813, 2774, 3543, 1657, 1696, 1579, 1795, 1865, 3503, 2948, 2898, 2845, 2785, 3533,
         1668, 1730, 1669, 3509, 2786])

#### FLMAE vertices -- to -- NPHM anchors 39
nphm_ind_39 = np.array([2712, 1579, 3485, 3756, 3430, 3659, 2711, 1575,  338,   27, 3631,
       3832, 2437, 1175, 3092, 2057, 3422, 3649, 3162, 2143,  617,   67,
       3172, 2160, 2966, 1888, 1470, 2607, 1896, 2981, 3332, 3231, 3494,
       3526, 3506, 3543, 3516, 3786, 3404])


ANCHOR_iBUG68_pairs_39 = np.array([
                        [0, 48], #left mouth corner
                        [1, 54], #right mouth corner
                        # [35, 50], # upper middle lip
                        # [35, 52], # upper middle lip
                        [35, 51], # upper middle lip
                        [34, 57], # lower middle lip
                        [38, 8], # chin
                        [33, 30], # nose
                        [10, 39], # left eye middle corner
                        [12, 36], # left eye outer corner
                        [11, 42], # rigth eye middle corner
                        [13, 45], # right corner outer corner
                        [8, 19], # left eyebrow middle
                        [9, 24], # right eye brow
                    ])


lables = ['mouth_corner_l',
            'mouth_corner_r',
            'jaw_nose_l',
            'jaw_nose_r',
            'below_mouth_left',
            'below_mouth_right',
            'cheek_left',
            'cheek_right',
            'eyebrow_left',
            'eyebrow_right',
            'eye_inner_left',
            'eye_inner_right',
            'eye_outer_left',
            'eye_outer_right',
            'nose_left',
            'nose_right',
            'jaw_left',
            'jaw_right',
            'temple_left',
            'temple_right',
            'ear_left',
            'ear_right',
            'forehead_left',
            'forhead_right',
            'top_head_left',
            'top_head_right',
            'back_head_left',
            'back_head_right',
            'back_neck_left',
            'back_neck_right',
            'shoulder_left',
            'shoulder_right',


            'throat',
            'nose_tip',
            'lower_lip',
            'upper_lip',
            'nose_top',
            'forehead',
            'chin',
            'rest'
            ]


MEDIA_PIPE_MOUTH_EYES_index = (61, 291,  # mouth corners
            133, 362,  # inner eye
            33, 263,  # outer eye
            15, 11) # lips
FLAME_MOUTH_EYES_index = (0, 1, 10, 11, 12, 13, 34, 35) # lips middlef


