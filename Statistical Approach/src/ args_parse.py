from tqdm import tqdm
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='ADR classfication project')

    parser.add_argument("--mode", type=str, default="train",
                        help="set a module in training or prediction mode")
    parser.add_argument("--arch", type=str, default="GB",
                        help="Classfier we use ")

    parser.add_argument("--model_name", type=str, default="GB",
                        help="type of model to load")
    parser.add_argument("--data_path", type=str, default="../data/",
                        help="type of model to load")
    parser.add_argument("--model_path", type=str, default="../models/",
                        help="type of model to load")
    parser.add_argument("--logs_path", type=str, default="../logs/",
                        help="type of model to load")
    parser.add_argument("--results_path", type=str, default="../figure/",
                        help="type of model to load")
    parser.add_argument("--figure_path", type=str, default="../figure/",
                        help="type of model to load")


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    if args.mode=="train":
        print("Hey its training time")
