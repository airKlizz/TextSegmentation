import argparse
from utils import create_data_wikinews

parser = argparse.ArgumentParser()
parser.add_argument("--num", help='Number of wikinews articles to use. The final number of data is less than this number because certain articles are depreciated', type=int, default=100)
parser.add_argument("--output", help='Name of the output file', type=str, default='wikinews.data.jsonl')

args = parser.parse_args()

create_data_wikinews(args.num, args.output)