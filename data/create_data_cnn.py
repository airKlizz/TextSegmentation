import argparse
from utils import create_data_cnn

parser = argparse.ArgumentParser()
parser.add_argument("--directory", help='Directory containing stories', type=str, default='stories/')
parser.add_argument("--output", help='Name of the output file', type=str, default='cnn.data.jsonl')

args = parser.parse_args()

create_data_cnn(args.directory, args.output)