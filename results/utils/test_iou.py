"""Program to test the IoU algorithm that is used as the reward function for the GraphQL environment.

It is similar to the SQL IoU algorithm but without the scaling part.
The scaling part did not give the right results when both the strings were exactly the same and hence it was removed.

This algorithm can be used to find the IoU for any 2 strings.
"""

from collections import Counter
from itertools import chain, groupby
from operator import itemgetter
from typing import Dict, List

def get_intersect_items(my_list: List, my_dict: Dict) -> List:
        """
        Returns the intersection of a list and a dictionary.
        """
        result = []
        for item in my_list:
            if item in my_dict:
                my_dict[item] -= 1
                if my_dict[item] == 0:
                    del my_dict[item]
                result.append(item)
        return result

def iou(str1: str, str2: str):
    AGENT_OBS = "agent_obs"
    EVAL_OBS = "eval_obs"
    REWARD = "reward"

    info = {}
    info[AGENT_OBS] = str1
    info[EVAL_OBS] = str2

    if isinstance(info[AGENT_OBS], list) and isinstance(info[EVAL_OBS], list):
        # Stringify all tuples to avoid comparator bugs and remove spaces
        list_agent = [str(x) for x in info[AGENT_OBS].replace(" ", "")]
        list_eval = [str(x) for x in info[EVAL_OBS].replace(" ", "")]

        # Reward: Intersection over Union
        dist_agent = Counter(list_agent)
        dist_eval  = Counter(list_eval)
        intersection = dist_agent & dist_eval

        get_key, get_val = itemgetter(0), itemgetter(1)
        merged_data = sorted(chain(dist_agent.items(), dist_eval.items()), key=get_key)
        union = {k: max(map(get_val, g)) for k, g in groupby(merged_data, key=get_key)}

        if len(union) == 0:
            # Outputs are both empty
            info[REWARD] = 1
        else:
            total_intersect = sum([v for _, v in intersection.items()])
            total_union = sum([v for _, v in union.items()])
            info[REWARD] = total_intersect * 1. / total_union
    else:
        info[REWARD] = 0.0

    return info[REWARD]

test_str_1 = "Error executing GraphQL query: Field 'UpdateUserInput.nodeId' of required type 'ID!' was not provided."
test_str_2 = "mutation {  updateUserById(input : { id: 3, userPatch: { username: \"Zack\"}}) {    user {      id,      username    }  }}"

print(iou(test_str_1, test_str_2))
