# type: ignore
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import models
from efficientnet_pytorch import EfficientNet
import seaborn as sns


def bench(model, n=5000, bs=1, cuda=True):   
    print(f"n: {n}, bs: {bs}, cuda: {cuda}")
    start = time.time()
    with torch.no_grad():
        for i in range(0, n, bs):
            print(i)
            bs = min(bs, n - i)
            data = torch.from_numpy(np.asarray(np.random.normal(size=(bs, 3, 224, 224)), dtype=np.float32))
            if cuda:
                data = data.cuda()
            res = model(data)
    return (time.time() - start) / n


def bench_batches(model, batches, n=100, cuda=True):    
    model.eval()
    if cuda:
        model.cuda()
    times = []
    try:
        for b in batches:
            times.append(bench(model, n=n, bs=b, cuda=cuda))
    except RuntimeError:
        print("Out of memory")
    return times


def get_model(args):
    if args.size.startswith("b"):
        model = EfficientNet.from_pretrained(f"efficientnet-{args.size}", advprop=args.adv_train)
        model_name = f"efficientnet-{args.size}_{'cuda' if args.cuda else 'cpu'}_{'adv_train' if args.adv_train else ''}"

    elif args.size.startswith("r"):
        resnets = {
            "r18": models.resnet18,
            "r34": models.resnet34,
            "r50": models.resnet50,
            "r101": models.resnet101,
            "r152": models.resnet152
        }

        model = resnets[args.size](pretrained=True)
        model_name = f"resnet-{args.size[1:]}_{'cuda' if args.cuda else 'cpu'}"

    return model, model_name


def bench_and_save_csv(args):
    batch_sizes = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 26]
    # batch_sizes = [1, 2, 3]

    model, model_name = get_model(args)

    times = bench_batches(model, batch_sizes, n=100, cuda=args.cuda)
    align = min(len(times), len(batch_sizes))

    df = pd.DataFrame()
    df["time"] = times[:align]
    df["batch_size"] = batch_sizes[:align]

    df["model"] = model_name

    df.to_csv(model_name + "_bench.csv")


def plot(log_csv_files):    
    dfs = []
    for fn in log_csv_files:
        dfs.append(pd.read_csv(fn))

    data = pd.concat(dfs)

    sns.set_theme(style="darkgrid")
    plot = sns.lineplot(data=data, x="batch_size", y="time", hue="model")
    plot.set_ylim(bottom=0)
    plot.set_title("Influence of batch size on inference time")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_files", nargs="*", type=str, required=False)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--adv_train", action="store_true")
    parser.add_argument("--size", type=str)

    args = parser.parse_args()

    if args.bench_files:
        plot(args.bench_files)    
    else:
        bench_and_save_csv(args)  
