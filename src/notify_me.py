
from utils import send_line_message
import argparse

def parse_arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, help="insert message")
    args = parser.parse_args()

    return args

def main() :
    args = parse_arguments()
    send_line_message(args.message)


if __name__ == "main" :
    main()


