import sys
import numpy as np
import pandas as pd
import os
import psutil
import csv
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth


def Apriori(ds, thold, m, num, rt, pc, o, pf):
    # read dataset
    data = pd.read_csv(ds, header=None)
    data = pd.DataFrame(data)

    # set a column called 'itemsets' for items
    data.columns = ['itemsets']

    # removing extra white space from dataset
    data['itemsets'] = data['itemsets'].str.strip()

    # splitting the items in the item-set separated by white space(' ')
    # convert into a list. apriori() takes a list and min_support as argument
    data['itemsets'] = data.itemsets.apply(lambda x: x.split(' '))

    # Encoding database transaction data in form of a Python list of lists into a NumPy array.
    te = TransactionEncoder()
    te_ary = te.fit(data['itemsets']).transform(data['itemsets'])
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # for starting time
    start = time.process_time()

    # apriori algorithm
    ap_patterns = apriori(df, min_support=float(thold), use_colnames=True)

    # calculate memory consumption
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / (1024 * 1024)

    # for ending time
    end = time.process_time()
    runtime = 1000 * (end - start)

    # calculate number of patterns
    number_of_patterns = len(ap_patterns)

    # parameter check
    if m:
        print("Memory usage: ", memory, "mb")
    if num:
        print("Number of frequent patterns: ", number_of_patterns)
    if rt:
        print("Runtime: ", runtime, "ms")
    if pc:
        print("Frequent patterns: \n", ap_patterns)
    if o:
        # output csv file generation
        output_file_name = algoName + '_' + ds + '.csv'
        exist = os.path.exists(output_file_name)
        if exist:
            data = [float(thold), runtime, memory]
            with open(output_file_name, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(data)
        else:
            data = [float(thold), runtime, memory]
            with open(output_file_name, 'w+', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(data)

    if pf:
        # pattern txt file generation

        pattern_file_name = algoName + '_' + ds + '_' + str(thold) + '.txt'
        pt_file = open(pattern_file_name, "w+")
        str_patterns = ap_patterns.to_string(header=True, index=True)
        pt_file.write(str_patterns)
        pt_file.close()


def FPgrowth(ds, thold, m, num, rt, pc, o, pf):
    # read dataset
    data = pd.read_csv(ds, header=None)
    data = pd.DataFrame(data)

    # set a column called 'itemsets' for items
    data.columns = ['itemsets']

    # removing extra white space from dataset
    data['itemsets'] = data['itemsets'].str.strip()

    # splitting the items in the item-set separated by white space(' ')
    data['itemsets'] = data.itemsets.apply(lambda x: x.split(' '))

    # Encoding database transaction data in form of a Python list of lists into a NumPy array.
    te = TransactionEncoder()
    te_ary = te.fit(data['itemsets']).transform(data['itemsets'])
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # for starting time
    start = time.process_time()

    # FP growth algorithm
    fp_patterns = fpgrowth(df, min_support=float(thold), use_colnames=True)

    # calculate memory consumption
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / (1024 * 1024)

    # for ending time
    end = time.process_time()
    runtime = 1000 * (end - start)

    # calculate number of patterns
    number_of_patterns = len(fp_patterns)

    # parameter check
    if m:
        print("Memory usage: ", memory, "mb")
    if num:
        print("Number of frequent patterns: ", number_of_patterns)
    if rt:
        print("Runtime: ", runtime, "ms")
    if pc:
        print("Frequent patterns: \n", fp_patterns)
    if o:
        # output csv file generation
        output_file_name = algoName + '_' + ds + '.csv'
        exist = os.path.exists(output_file_name)
        if exist:
            data = [float(thold), runtime, memory]
            with open(output_file_name, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(data)
        else:
            data = [float(thold), runtime, memory]
            with open(output_file_name, 'w+', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(data)

    if pf:
        # pattern txt file generation

        pattern_file_name = algoName + '_' + ds + '_' + str(thold) + '.txt'
        pt_file = open(pattern_file_name, "w+")
        str_patterns = fp_patterns.to_string(header=True, index=True)
        pt_file.write(str_patterns)
        pt_file.close()


# main function

if __name__ == '__main__':
    n = len(sys.argv)
    # print(n)
    algoName = "AP"
    dataset = "Toy.txt"
    threshold = 0.2
    m = False
    num = False
    rt = False
    pc = False
    o = False
    pf = False
    if n > 1:
        for i in range(1, n):
            parameter = sys.argv[i]

            if parameter == "-a":
                algoName = sys.argv[i + 1]
                if algoName != "AP" and algoName != "FP":
                    print("Wrong input")

            if parameter == "-d":
                dataset = sys.argv[i + 1]
                if dataset != "mushroom.dat" and dataset != "chess.dat" and dataset != "kosarak.dat" and dataset != "retail.dat" and dataset != 'Toy.txt':
                    print("Wrong input")
            if parameter == "-t":
                threshold = sys.argv[i + 1]
            if parameter == "-m":
                m = True
            if parameter == "-n":
                num = True
            if parameter == "-rt":
                rt = True
            if parameter == "-pc":
                pc = True
            if parameter == "-o":
                o = True
            if parameter == "-pf":
                pf = True

    if algoName == "AP":
        Apriori(dataset, threshold, m, num, rt, pc, o, pf)
    if algoName == "FP":
        FPgrowth(dataset, threshold, m, num, rt, pc, o, pf)
