#!/usr/bin/env python

import sys

def print_status_bar(i, progress, N, size=50):
    if i % (N/size) == 0:
        progress += 1

    sys.stdout.write('\r')
    sys.stdout.write("Status: [" + "="*progress + " "*(size-progress) +  "]")
    sys.stdout.flush()

    return progress

def print_execution_time(begin, end):
    total_calculation_time = float((end - begin).total_seconds())
    print("Calculation time: %f seconds" % total_calculation_time)