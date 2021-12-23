import sys
import numpy as np
import pandas as pd
import pyfpgrowth
import os
import psutil



def Apriori(ds, thold, m):
   # print("Algo : Apriori")
   # print("Dataset: ", ds)
    #print("Threshold: ", thold)
    if m:
        process = psutil.Process(os.getpid())
        print("Memory usage: ", process.memory_info().rss * (1 / 1000000))


def FPgrowth(ds, thold, m):

    # read dataset
    data = pd.read_csv(ds, header=None)
    data = pd.DataFrame(data)

    # set a column called 'itemsets' for itemsets
    data.columns = ['itemsets']

    # splitting the items in the item-set by space(' ')
    # convert into a list. pyfpgrowth.frequent_patterns() takes a list and min_support as argument
    data['itemsets'] = data.itemsets.apply(lambda x: x.split(' '))

    # min_support conversion
    rows = len(data)
    tl = float(rows * float(thold))
    print("minimum support: ", tl)
    patterns = pyfpgrowth.find_frequent_patterns(data['itemsets'], tl)

    rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
    print(patterns)
    if m:
        process = psutil.Process(os.getpid())
        print("Memory usage: ", process.memory_info().rss * (1 / 1000000))


if __name__ == '__main__':
    n = len(sys.argv)
    print(n)
    algoName = "AP"
    dataset = "Toy.txt"
    threshold = 0.2
    m=False

    if n > 1:
        for i in range(1, n):
            parameter = sys.argv[i]

            if parameter == "-a":
                algoName = sys.argv[i + 1]
                if algoName != "AP" and algoName != "FP":
                    print("Wrong input")

            if parameter == "-d":
                dataset = sys.argv[i + 1]

            if parameter == "-t":
                threshold = sys.argv[i + 1]
            if parameter == "-m":
                m = True

    if algoName == "AP":
        Apriori(dataset, threshold, m)
    if algoName == "FP":
        FPgrowth(dataset, threshold, m)
