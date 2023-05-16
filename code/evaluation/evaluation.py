"""
Evaluate subtask 1 of SEPP-NLG 2021: https://sites.google.com/view/sentence-segmentation/
"""

__author__ = "don.tuggener@zhaw.ch"

import os
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy


def display_confusion_matrix(cm, file_name, subtask):
    # provide a confusion matrix showing expected and predicted break for each classes
    plt.clf()
    cmn = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    target_names = ['Abs. pauses', 'Pauses C', 'Pauses M', 'Pauses L'] if subtask == "break_size" else ['Abs. pauses', 'Pauses']
    ax = sns.heatmap(cmn, linewidths=1,  annot=True, fmt='.2f', cmap=plt.cm.Blues, xticklabels=target_names, yticklabels=target_names)
    ax.set_xlabel('Pauses prédites')
    ax.set_ylabel('Pauses attendues')
    plt.savefig(file_name)


def get_scores(well_found, almost_found, quite_found, not_found, over_detect):
    # method that weights the types of errors
    total = well_found + almost_found + quite_found + not_found
    almost_found = almost_found * 0.75
    quite_found = quite_found * 0.5
    precision = (well_found + almost_found + quite_found) / (well_found + almost_found + quite_found + over_detect)
    rappel = (well_found + almost_found + quite_found) / (well_found + almost_found + quite_found + not_found)
    f_measure = 2 * (precision*rappel) / (precision+rappel)
    print(str(round(precision, 2))+"\t" + str(round(rappel, 2))+"\t" + str(round(f_measure, 2))+"\t" + str(total))
    return precision, rappel, f_measure


def get_detail_results(all_gt_labels, all_predicted_labels):
    # method that calculate and print new scores with personalized weight according to the importance of the error
    i = lab0_well_found = lab0_not_found = lab1_well_found = lab1_almost_found = lab1_quite_found = lab1_not_found = \
        lab2_well_found = lab2_almost_found = lab2_not_found = lab3_well_found = lab3_almost_found = \
        lab3_quite_found = lab3_not_found = lab0_over_detect = lab1_over_detect = lab2_over_detect = \
        lab3_over_detect = 0
    for predicted_label in all_predicted_labels:
        gt_label = all_gt_labels[i]
        i += 1
        if gt_label == '0':
            if predicted_label == '0':
                lab0_well_found += 1
            elif predicted_label == '1':
                lab0_not_found += 1
                lab1_over_detect += 1
            elif predicted_label == '2':
                lab0_not_found += 1
                lab2_over_detect += 1
            elif predicted_label == '3':
                lab0_not_found += 1
                lab3_over_detect += 1
        elif gt_label == '1':
            if predicted_label == '1':
                lab1_well_found += 1
            elif predicted_label == '2':
                lab1_almost_found += 1
                lab2_over_detect += 1
            elif predicted_label == '3':
                lab1_quite_found += 1
                lab3_over_detect += 1
            elif predicted_label == '0':
                lab1_not_found += 1
                lab0_over_detect += 1
        elif gt_label == '2':
            if predicted_label == '2':
                lab2_well_found += 1
            elif predicted_label == '1':
                lab2_almost_found += 1
                lab1_over_detect += 1
            elif predicted_label == '3':
                lab2_almost_found += 1
                lab3_over_detect += 1
            elif predicted_label == '0':
                lab2_not_found += 1
                lab0_over_detect += 1
        elif gt_label == '3':
            if predicted_label == '3':
                lab3_well_found += 1
            elif predicted_label == '2':
                lab3_almost_found += 1
                lab2_over_detect += 1
            elif predicted_label == '1':
                lab3_quite_found += 1
                lab1_over_detect += 1
            elif predicted_label == '0':
                lab3_not_found += 1
                lab0_over_detect += 1

    lab0_p, lab0_r, lab0_fm = \
        get_scores(lab0_well_found, 0, 0, lab0_not_found, lab0_over_detect)
    print("P0: " + str(lab0_p) + " - R0: " + str(lab0_r) + " - FM0: " + str(lab0_fm))
    lab1_p, lab1_r, lab1_fm = \
        get_scores(lab1_well_found, lab1_almost_found, lab1_quite_found, lab1_not_found, lab1_over_detect)
    print("P1: " + str(lab1_p) + " - R1: " + str(lab1_r) + " - FM1: " + str(lab1_fm))
    lab2_p, lab2_r, lab2_fm = get_scores(lab2_well_found, lab2_almost_found, 0, lab2_not_found, lab2_over_detect)
    print("P2: " + str(lab2_p) + " - R2: " + str(lab2_r) + " - FM2: " + str(lab2_fm))
    lab3_p, lab3_r, lab3_fm = \
        get_scores(lab3_well_found, lab3_almost_found, lab3_quite_found, lab3_not_found, lab3_over_detect)
    print("P3: " + str(lab3_p) + " - R3: " + str(lab3_r) + " - FM3: " + str(lab3_fm))


def evaluate_subtask(gold_path: str, predicted_path: str, subtask: str) -> None:
    all_gt_labels, all_predicted_labels = (list(), list(),)  # aggregate all labels over all files
    file_names = os.listdir(gold_path)
    for i, file_name in enumerate(file_names, 1):
        # get ground truth labels
        with open(os.path.join(gold_path, file_name), "r", encoding="utf8",) as f:
            lines = f.read().strip().split("\n")
            rows = [line.split(",") for line in lines[1:]]
            if subtask == "punct2break":
                for row in rows:
                    if row[2] == "1" or row[2] == "2" or row[2] == "3":
                        row[2] = '1'
            gt_labels = [(row[1] if subtask == "break" else row[2]) for row in rows]

        # get predicted labels
        with open(os.path.join(predicted_path, file_name), "r", encoding="utf8",) as f:
            lines = f.read().strip().split("\n")
            rows = [line.split("\t") for line in lines]
            if subtask == "punct2break":
                for row in rows:
                    if row[2] == "." or row[2] == ";" or row[2] == "," or row[2] == "?" or row[2] == "!":
                        row[2] = '1'
                    else:
                        row[2] = '0'
        predicted_labels = [(row[1] if subtask == "break" else row[2]) for row in rows]

        assert len(gt_labels) == len(predicted_labels), \
            f"unequal no. of labels for files {gt_labels} & {predicted_labels}"
        all_gt_labels.extend(gt_labels)
        all_predicted_labels.extend(predicted_labels)

    # get Precision/Recall/F-Measure
    eval_result = classification_report(all_gt_labels, all_predicted_labels)
    print(eval_result)

    # --- get confusion matrix ---
    labels = ['0', '1', '2', '3'] if subtask == "break_size" else ['0', '1']
    cm = confusion_matrix(all_gt_labels, all_predicted_labels, labels=labels, normalize='true')
    display_confusion_matrix(cm, subtask+'tmp.pdf', subtask)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluation of the break detection system")
    parser.add_argument("--gold_path", help="dataset gold", default="data/orfeo-synpaflex/test/")
    parser.add_argument("--predicted_path", help="dataset to evaluate", default="results/orfeo-synpaflex_pauzee")
    parser.add_argument("--subtask",
                        help='break or break_size --- note : to evaluate pauzee in break prediction use the break '
                             'subtask, to evaluate unbabel: please, use the punct2break function to convert '
                             'punctuation to break. The break_size subtask allows the evaluation of Pauzee in break '
                             'length prediction',
                        default="break")
    args = parser.parse_args()
    evaluate_subtask(args.gold_path, args.predicted_path, args.subtask)
