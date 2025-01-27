import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Magazine')
parser.add_argument('--data_name', type=str, default='Magazine_5')
parser.add_argument('--review_rate', type=str, default='0.7')
parser.add_argument('--fine_tune', type=str, default="False")
parser.add_argument('--bz', type=int, default=4)
parser.add_argument('--with_summary', action='store_true')
parser.add_argument('--mode', type=str, default="ORIG")

args = parser.parse_args()


