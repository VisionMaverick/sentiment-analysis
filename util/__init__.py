# This file is using encoding: utf-8
import time
import csv

def get_time_stamp():
    return time.strftime("%y%m%d-%H%M%S-%Z")


def extract_pos_neg_data(file_name, delimi, quochar, condition_posi, condition_neg, condi_idx, col_idx_to_extract):
    # read all data and segregate in two files for positive and negative review
    fp = open(file_name, 'rt', encoding='utf-8')

    reader = csv.reader(fp, delimiter=delimi, quotechar=quochar, escapechar='\\')

    fq_pos = open('positive_review', 'w')
    fq_neg = open('negative_review', 'w')

    # treat neutral and irrelevant the same
    for row in reader:
        if row[condi_idx] == condition_posi:
            pos_review = row[col_idx_to_extract]
            fq_pos.write(pos_review)
        elif row[condi_idx] == condition_neg:
            neg_review = row[col_idx_to_extract]
            fq_neg.write(neg_review)
    fp.close()
    fq_pos.close()
    fq_neg.close()

    return 0