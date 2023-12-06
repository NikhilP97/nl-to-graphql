"""Program to run any of the environments in an interactive command line mode, supports human and LLM policies.

Human policy means users will have to type in technical commands themselves.
Example - For the GraphQL environment, a valid command would be "{   allPosts {     nodes {       title       body     }   } }" to get the title and body from the post table.

LLM policy means the user can type the question in natural language, then this is passed to a LLM (GPT-3.5). The response from the LLM will be a valid technical statement.
For the GraphQL environment, this would be "{   allPosts {     nodes {       title       body     }   } }"

All technical statement are then executed on their requisite engines. In the case of GraphQL environment, it would be the GraphQL server that is exposed through the docker-compose files.

This file is copy of run_demo.py and has been modified to support the above 2 policies.
run_demo.py only supports human policy.

The file does not support a lot of variables through the CLI. If you need to make any changes like changing the LLM, make modifications directly in this file or
add support for it through the command line arguments. 

Typical usage example:

    Generic:
    python run_demo_human_and_llm.py {environment_name} --policy {human|llm}

    Human policy with GraphQL
    python run_demo_human_and_llm.py graphql --policy human

    LLM policy with GraphQL
    python run_demo_human_and_llm.py graphql --policy llm

    Human policy with SQL
    python run_demo_human_and_llm.py sql --policy human

    LLM policy with BASH
    python run_demo_human_and_llm.py bash --policy llm
"""
import argparse

from intercode.envs import (
    BashEnv, PythonEnv, SqlEnv, CTFEnv, SWEEnv, GraphQLEnv
)
from experiments.policies import (
    HumanPolicy, ChatGPTPolicy
)
from typing import Dict, List


def preprocess_ctf(record: Dict) -> List:
    cmds = [f"cd /ctf/{record['task_id']}"]
    if "setup" in record:
        cmds.append(record["setup"])
    return cmds

def preprocess_sql(record: Dict) -> List:
    db = record["db"]
    return [f"use {db}"]

DEMO_MAP = {
    "graphql": {"env": GraphQLEnv, "image_name": "blog-graphql", "data_path": "./data/graphql/nl2graphql_fs_1.json", "schema_path": "./data/graphql/nl2graphql_fs_1.graphql"},
    "bash": {"env": BashEnv, "image_name": "intercode-nl2bash", "data_path": "./data/nl2bash/nl2bash_fs_1.json"},
    "python": {"env": PythonEnv, "image_name": "intercode-python", "data_path": "./data/python/mbpp/ic_mbpp.json"},
    "sql": {"env": SqlEnv, "image_name": "docker-env-sql", "data_path": "./data/sql/bird/ic_bird.json", "preprocess": preprocess_sql},
    "ctf": {"env": CTFEnv, "image_name": "intercode-ctf", "data_path": "./data/ctf/ic_ctf.json", "preprocess": preprocess_ctf},
    "swe": {"env": SWEEnv, "image_name": "intercode-swe", "data_path": "./data/swe-bench/ic_swe_bench.json"}
}

SETTING_MAP = {
    "sql": "MySQL Database",
    "bash": "Bourne Shell",
    "python": "Python 3 Interpreter",
    "ctf": "Capture the Flag",
    "graphql": "GraphQL Server"
}

def main_human(demo: str, policy: str):
    if demo not in DEMO_MAP:
        raise ValueError(f"Demo {demo} not supported (Specify one of [bash, python, sql, graphql])")
    image_name = DEMO_MAP[demo]["image_name"]
    data_path = DEMO_MAP[demo]["data_path"] if "data_path" in DEMO_MAP[demo] else None
    preprocess = DEMO_MAP[demo]["preprocess"] if "preprocess" in DEMO_MAP[demo] else None

    env = DEMO_MAP[demo]["env"](image_name, data_path=data_path, verbose=True, preprocess=preprocess)
    
    try:
        for _ in range(3):
            env.reset()
            policy = HumanPolicy()
            obs = env.observation
            done = False
            query = env.query if hasattr(env, "query") else None

            while not done:
                action = policy.forward(query, obs, env.get_available_actions())
                obs, reward, done, info = env.step(action)
    except KeyboardInterrupt:
        print("Exiting InterCode environment...")
    finally:
        env.close()

def main_llm(demo: str, policy: str):
    if demo not in DEMO_MAP:
        raise ValueError(f"Demo {demo} not supported (Specify one of [bash, python, sql, graphql])")
    
    image_name = DEMO_MAP[demo]["image_name"]
    data_path = DEMO_MAP[demo]["data_path"] if "data_path" in DEMO_MAP[demo] else None
    schema_path = DEMO_MAP[demo]["schema_path"] if "schema_path" in DEMO_MAP[demo] else None
    
    schema_value = ""
    if schema_path:
        with open(schema_path) as f:
            schema_value = f.read()
    
    preprocess = DEMO_MAP[demo]["preprocess"] if "preprocess" in DEMO_MAP[demo] else None
    env = DEMO_MAP[demo]["env"](image_name, data_path=data_path, verbose=True, preprocess=preprocess)
    
    try:
        for _ in range(3):
            env.reset()
            human_policy = HumanPolicy()
            chat_gpt_policy = ChatGPTPolicy(language=demo, setting=SETTING_MAP[demo],
                template="query", dialogue_limit="1", model="gpt-3.5-turbo", schema=schema_value)
            obs = env.observation
            done = False
            query = env.query if hasattr(env, "query") else None

            while not done:
                obs, reward = None, None
                input_natural_language = human_policy.forward(query, obs, env.get_available_actions())
                print(f"Input natural language: {input_natural_language} \n")

                try:
                    chat_gpt_policy.reset()
                    action, is_code = chat_gpt_policy.forward(
                        input_natural_language,
                        obs,
                        reward,
                        env.get_available_actions())
                except (ValueError, TypeError) as e:
                    print(f"[ERROR] while invoking chatGPT: {e}")
                    break

                print(f"Response from LLM: : {action} \n")
                obs, reward, done, info = env.step(action)
    except KeyboardInterrupt:
        print("Exiting InterCode environment...")
    finally:
        env.close()


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("demo", type=str, help="Environment demo to run [bash, python, sql]")
    argparse.add_argument('--policy', choices=['human', 'llm'], help='Evaluation policy for the input query')
    args = argparse.parse_args()

    if args.policy == "human":
        main_human(**vars(args))
    else:
        main_llm(**vars(args))