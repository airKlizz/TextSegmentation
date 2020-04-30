import argparse
from utils import create_train_test_data

parser = argparse.ArgumentParser()
parser.add_argument("--input", help='Path to the .jsonl data file', type=str, default='wikinews.data.jsonl')
parser.add_argument("--train_output", help='Name of the train output file', type=str, default='train.txt')
parser.add_argument("--test_data_output", help='Name of the test data output file', type=str, default='test.data.txt')
parser.add_argument("--test_gold_output", help='Name of the test gold output file', type=str, default='test.gold.txt')


args = parser.parse_args()

create_train_test_data(args.input, args.train_output, args.test_data_output, args.test_gold_output, test_size=0.2, random_state=2020)