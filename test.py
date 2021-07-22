import argparse

import pandas as pd
import torch

from scripts.model import calculate_acc, VerificationCodeCNN, build_dataloader


def add_arguments(parser):
    """Helper function to fill the parser object.
    Args:
        parser: Parser object
    Returns:
        None
    """
    parser.add_argument(
        "-w",
        "--weights",
        default="./data/model/codeocr_params.pkl",
        help="the file storage model weights",
        type=str)
    parser.add_argument(
        "-i",
        "--input",
        default="./data/testing",
        help="test folder",
        type=str)
    parser.add_argument(
        "-o",
        "--output", default="./data/result/prediction.csv",
        help="the file to save result",
        type=str)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="whether run at gpu")


def test(args):
    model = VerificationCodeCNN()
    print(f"use {'gpu' if args.gpu else 'cpu'}")
    model.load_state_dict(torch.load(args.weights, map_location='gpu' if args.gpu else "cpu"))

    dl_test = build_dataloader(args.input, batch_size=1024, shuffle=False)
    model.eval()
    with torch.no_grad():
        y_pred = torch.cat([model.forward(t[0]) for t in dl_test])
    y_true = torch.cat([t[1] for t in dl_test])
    acc, p, t = calculate_acc(y_pred, y_true, mode="predicting")
    res = pd.DataFrame(
        {
            "Code": t,
            "Predicted": p
        })
    res.to_csv(args.output, index=False)
    return res, acc


def main():
    PARSER = argparse.ArgumentParser()
    add_arguments(PARSER)
    args = PARSER.parse_args()
    res, acc = test(args)
    return res, acc


if __name__ == "__main__":
    res, acc = main()
    print(f"{res} \n Accuracy: {acc}")
