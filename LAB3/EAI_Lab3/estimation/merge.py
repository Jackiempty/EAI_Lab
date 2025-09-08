import numpy as np


def merge(subset, min_num_body_parts=4, min_score=0.4):
    """
    Estimates the skeletons.
    :param connections: valid connections
    :param min_num_body_parts: minimum number of body parts for a skeleton
    :param min_score: minimum score value for the skeleton
    :return: list of skeletons. Each skeleton has a list of identifiers of body parts:
        [
            [id1, id2,...,idN, score, parts_num],
            [id1, id2,...,idN, score, parts_num]
            ...
        ]

    position meaning:
        [   [nose       , neck           , right_shoulder , right_elbow      , right_wrist  , left_shoulder
             left_elbow , left_wrist     , right_hip      , right_knee       , right_ankle  , left_hip
             left_knee  , left_ankle     , right_eye      , left_eye         , right_ear    , left_ear
             score, parts_num],
        ]
    """

    # 2 step :
    #---merge----
    # Merge the limbs in the subset
    # score : score
    # parts_num : How many limbs are in the subset 
    ###############################
    
    # after merge
    #---delete---
    # Delete the non-compliant subset
    # 1. parts_num < 4
    # 2. Average score(score / parts_num) < 0.4 
    ############################################
    
    delete_idx = []
    for i in range(len(subset)): 
        if ?????????????? or ???????????????:  # revise here
            delete_idx.append(i)
    subset = np.delete(subset, delete_idx, axis=0)

    return subset   