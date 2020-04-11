import argparse
from utils import eval, print_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--gold_filename", help='Path to the gold test file', type=str, default='wikinews/test.gold.txt')
parser.add_argument("--candidate_filename", help='Path to the candidate test file', type=str, default='wikinews/test.run.txt')

args = parser.parse_args()

metrics = eval(args.gold_filename, args.candidate_filename)
print_metrics(metrics)