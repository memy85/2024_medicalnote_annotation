
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *
import logging
import argparse





def argparser() :

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path")
    parser.add_argument("--model-path")
    parser.add_argument("--loggig-path")
    parser.add_argument("--config-path")

    args = parser.parse_args()


def main() :


    pass


if __name__ == "__main__" :

    main()