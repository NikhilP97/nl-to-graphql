"""Class which has the core implementation of the GraphQL environment.

It implements the exec_action function that takes the GraphQL query and sends it to the GraphQL server.
The response from the GraphQL server is then stored as the observation in the class variable.

It also implements the get_reward function that determines the accuracy of the LLM generated result and the gold generated results.
The reward function calculates 2 values:
1. The IoU value of the GraphQL server response between the agent and the gold query.
1. The IoU value of the GraphQL query between the agent and the gold query.
"""

from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

from collections import Counter
from itertools import chain, groupby
from operator import itemgetter
from typing import Dict, List, Tuple

from intercode.envs.ic_env import (
    IntercodeEnv,
    AGENT_OBS, EVAL_OBS, CORRUPT_GOLD, ACTION_EXEC, REWARD, LLM_RESPONSE, GOLD, QUERY_REWARD
)

GRAPHQL_CONFIG = {
    'url': 'http://127.0.0.1:5433/graphql',
}

class GraphQLEnv(IntercodeEnv):
    """Gym environment for GraphQL"""
    name = "ic_graphql"

    def __init__(self, image_name: str, **kwargs):
        super(GraphQLEnv, self).__init__(image_name, **kwargs)

        # Establish connection with the GraphQL server
        self.transport = AIOHTTPTransport(url=f"{GRAPHQL_CONFIG['url']}")
        self.client = Client(transport=self.transport, fetch_schema_from_transport=True)

    def exec_action(self, action: str) -> None:
        try:
            self.action = action
            query = gql(action)
            # Execute the query on the transport
            result = self.client.execute(query)
            self.info[ACTION_EXEC] = True
            self.observation = result
        except Exception as err:
            self.observation = f"Error executing GraphQL query: {err}"
            self.info[ACTION_EXEC] = False
    
    def get_reward(self) -> Tuple[float, Dict]:
        """
        The reward currently is calculated as an intersection over union
        between the agent's answer and the gold answer.
        """
        self.info = {}
        self.info[AGENT_OBS] = self.observation
        gold_query_result = {}

        # Run gold command(s) in evaluation container
        try:
            gold_query_result = self.client.execute(gql(self.gold))
            self.info[CORRUPT_GOLD] = False
        except Exception as err:
            self.info[EVAL_OBS] = f"Error executing query: {err}"
            self.info[CORRUPT_GOLD] = True
        self.info[EVAL_OBS] = {}
        self.info[EVAL_OBS] = gold_query_result

        # Calculate IoU of gold query based GraphQL response and LLM query based GraphQL response
        if isinstance(self.info[AGENT_OBS], dict) and isinstance(self.info[EVAL_OBS], dict):
            # Stringify the output result and remove spaces
            list_agent = str(self.info[AGENT_OBS]).replace(" ", "")
            list_eval = str(self.info[EVAL_OBS]).replace(" ", "")

            # Reward: Intersection over Union
            dist_agent = Counter(list_agent)
            dist_eval  = Counter(list_eval)
            intersection = dist_agent & dist_eval

            get_key, get_val = itemgetter(0), itemgetter(1)
            merged_data = sorted(chain(dist_agent.items(), dist_eval.items()), key=get_key)
            union = {k: max(map(get_val, g)) for k, g in groupby(merged_data, key=get_key)}

            if len(union) == 0:
                # Outputs are both empty
                self.info[REWARD] = 1
            else:
                total_intersect = sum([v for _, v in intersection.items()])
                total_union = sum([v for _, v in union.items()])
                self.info[REWARD] = total_intersect * 1. / total_union

        else:
            self.info[REWARD] = 0.0

        self.reward = self.info[REWARD]

        # Cast observations to strings to avoid JSON serialization errors
        self.info[AGENT_OBS] = str(self.info[AGENT_OBS])
        self.info[EVAL_OBS] = str(self.info[EVAL_OBS])


        # Calculate IoU of the gold query and LLM generated query
        self.info[GOLD] = self.gold
        self.info[LLM_RESPONSE] = self.action
        if isinstance(self.info[GOLD], str) and isinstance(self.info[LLM_RESPONSE], str):
            # Stringify the output result and remove spaces
            list_agent = self.info[GOLD].replace(" ", "")
            list_eval = self.info[LLM_RESPONSE].replace(" ", "")

            # Reward: Intersection over Union
            dist_agent = Counter(list_agent)
            dist_eval  = Counter(list_eval)
            intersection = dist_agent & dist_eval

            get_key, get_val = itemgetter(0), itemgetter(1)
            merged_data = sorted(chain(dist_agent.items(), dist_eval.items()), key=get_key)
            union = {k: max(map(get_val, g)) for k, g in groupby(merged_data, key=get_key)}

            if len(union) == 0:
                # Outputs are both empty
                self.info[QUERY_REWARD] = 1
            else:
                total_intersect = sum([v for _, v in intersection.items()])
                total_union = sum([v for _, v in union.items()])
                self.info[QUERY_REWARD] = total_intersect * 1. / total_union

        else:
            self.info[QUERY_REWARD] = 0.0

        self.logger.info(f"Info: {self.info}")
        self.logger.info(f"Reward: {self.reward}")
        return self.reward, self.info

    def close(self):
        self.logger.info("Beginning environment shutdown...")
        self.container.stop()
        self.logger.info("Agent, evaluation containers stopped")
