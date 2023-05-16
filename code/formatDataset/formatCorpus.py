# coding: utf-8
# !/usr/bin/python
import os
import re
import pandas as pd
import numpy as np
import sys

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


# format corpus

def clean_line(w):
    w = re.sub(r" *<break(?P<ms>[0-9]+ms)> *", r" \g<ms> ", w)
    w = re.sub(r"(?P<punct>[\.\!\?;,:]+)", r" \g<punct> ", w)
    w = re.sub(r" +", " ", w)
    w = re.sub(r"^ +", "", w)
    return re.sub(r" +$", "", w)


def format_orfeo(file):
    output = ""
    ts_e_before, word_before = 0.000, ""  # orfeo
    for line in file.split("\n"):
        if line != "" and line[0] != "#":
            info = line.split("\t")
            if info[12] != "Public":
                word = info[1]
                ts_b = float(info[10])
                break_before = ts_b - ts_e_before
                break_before_ms = int(round(break_before * 1000, 0))
                if word_before != "":
                    output += word_before + "\t0\t0\t" + str(break_before_ms) + "ms\n"
                word_before, ts_e_before = word, float(info[11])
    return output


def format_synpaflex(file):
    output = ""
    for line in file.split("\n"):
        if line != '':
            line = line + "<break>"
            line = re.sub(r" +<", r"<", line)
            line = re.sub(r"<break ", r"<break", line)
            for w in line.split(" "):
                if w != "":
                    w = clean_line(w)
                    infos = w.split(" ")
                    word, punct, break_ms, end = infos[0], "0", "0", "0"
                    if len(infos) > 1:
                        if "ms" in infos[1]:
                            break_ms = infos[1]
                        else:
                            punct = infos[1]
                            if len(infos) > 2:
                                break_ms = infos[2]
                        if "." in punct or "?" in punct or "!" in punct:
                            end = "1"
                    output += (word + "\t" + end + "\t" + punct + "\t" + break_ms + "\n")
    return output


# format

def clean_ms(df):
    df['ms'] = df['ms'].replace(['ms'], '', regex=True)
    df['ms'] = df['ms'].replace(['<break>'], '-1000', regex=True)
    df["ms"] = pd.to_numeric(df["ms"])
    max_ms = df.loc[df["ms"].idxmax()]["ms"]
    df["ms"] = df["ms"].astype(str)
    df['ms'] = df['ms'].replace(['-1000'], max_ms, regex=True)
    df["ms"] = pd.to_numeric(df["ms"])
    return df


def get_threshold(all_ms):
    threshold, i = 0, 78
    while (threshold < 80) & (i < 84):
        i += 1
        threshold = np.percentile(all_ms, i)
    if threshold < 80:
        while (threshold < 50) & (i < 90):
            i += 1
            threshold = np.percentile(all_ms, i)
    if threshold < 10:
        i += 1
        threshold = np.percentile(all_ms, i)
    r = 100 - i
    s_i = i + (r * 0.3)  # 89 if i < 85 else i + 3 30%
    m_i = i + (r * 0.66)  # 94 if s_i < 92 else s_i + 5 66%
    # print(str(i)+" - "+str(s_i)+" - "+str(m_i))
    # print(str(threshold)+" \t - PC -"+str(np.percentile(all_ms, s_i))+"\t - PM - "+str(np.percentile(all_ms, m_i)))
    return threshold, np.percentile(all_ms, s_i), np.percentile(all_ms, m_i)


def is_break(ms, break_threshold):
    if ms < break_threshold:
        return 0
    else:
        return 1


def get_break_cat(ms, break_threshold, small_break_threshold, medium_break_threshold):
    if ms < break_threshold:
        return "0"
    elif ms < small_break_threshold:
        return "1"
    elif ms < medium_break_threshold:
        return "2"
    else:
        return "3"


def format_file(file, data_type):
    # format lines :
    formated_text = ""
    if data_type == "orfeo":
        formated_text = format_orfeo(file)
    elif data_type == "synpaflex":
        formated_text = format_synpaflex(file)

    # break inference - short - medium - long break
    df = clean_ms(pd.read_csv(StringIO(formated_text), sep="\t", header=None, names=["word", "end", "punct", "ms"]))
    # add percentiles information
    break_threshold, small_break_threshold, medium_break_threshold = get_threshold(list(df['ms']))
    df['isBreak'] = df['ms'].apply(lambda x: is_break(x, break_threshold))
    df['catBreak'] = df['ms'].apply(
        lambda x: get_break_cat(x, break_threshold, small_break_threshold, medium_break_threshold))
    df = df.drop(['ms'], axis=1)
    df = df.drop(['end'], axis=1)
    df = df.drop(['punct'], axis=1)
    return df


def file_status_orfeo(file_name):
    file_name = file_name.replace(".orfeo", "")
    test_list = ["Bloch_052-2_ZIBELINE", "Boyer_044-2_LEGENDE_CREATION", "Boyer_056-4_FERAON",
                 "Boyer_076-2_HISTOIRE_DE_FEMMES", "Buleon_044-1_CONTE_DE_LA_CREATION", "Buleon_055-8_LES_VOMISSURES",
                 "Calandry_038-4_LE_CRAPAUD_ET_LA_TORTUE", "Calandry_042-6_LES_TROIS_CHEVEUX_DOR",
                 "Calandry_056-3_JUSTICE_ET_INJUSTICE", "Caudal_LANKOU_2", "Cevin_060-2_LE_CORBEAU_ET_LA_PRINCESSE",
                 "De_La_Rochefoucauld_MELUSINE", "De_La_Salle_GRAND-MERE_MENSONGE_1",
                 "De_La_Salle_GRAND-MERE_MENSONGE_2", "Desnouveaux_049-4_NATIVITE_DE_LA_VIERGE_BIBLIQUE",
                 "Garrigue_083-2_LALA_AHISHA", "Guillemin_061-5_LA_COMPAGNIE_DES_LOUPS", "Guillemin_080-4_ZOBEIDA",
                 "Kiss_202i-6_ARUN_AL_RACHID", "Kiss_202i-12-13_HELENE_LA_MAGIQUE", "Kiss_205-4_HORO_LE_GUERRIER",
                 "Kiss_205-5_NIKORIMA", "Mastre_106-2_CONTE_DE_BOURGOGNE_2", "Nataf_041-4_LA_CHEMISE_MAGIQUE",
                 "Quere_063-3_1001_NUITS_3", "Sauvage_071-1_LE_ROI_DES_PIGEONS", "Sauvage_071-2_LE_COQ_ET_LE_RENARD",
                 "Sauvage_071-3_LE_CORBEAU-LE_RAT_ET_LA_TORTUE", "Sauvage_071-5_LE_BRAHMANE_ET_SA_FEMME",
                 "Sauvage_071-6_IRAGNAKA_2", "Walerski_036-2_LA_PIERRE_BARBUE"]
    train_list = ["Bizouerne_036-4_CELUI_QUI_NE_VEUT_PAS_MOURIR_2", "Bizouerne_037-1_CONTE_JUIF",
                  "Bizouerne_039-2_OUNAMIR", "Bizouerne_039-4_POKOU_1", "Bizouerne_039-6_JEUNE_HOMME",
                  "Bizouerne_039-8_CELUI_QUI_NE_VEUT_PAS_MOURIR_1", "Bizouerne_047-3_PYGMALION",
                  "Bizouerne_055-5_POKOU_2", "Bizouerne_061-7_LE_VAMPIRE", "Bizouerne_062-2_BARBIER",
                  "Bizouerne_062-3_LE_MARI_TROMPE", "Bizouerne_062-5_AVEUGLE", "Bizouerne_062-9_BARBIER_FIN",
                  "Bloch_052-3-CHEVAL_DE_MARBRE", "Bloch_052-5_DEUX_ECOSSAIS", "Boyer_069-3_CORDONNIER",
                  "Boyer_083-3_SAVETIER", "Boyer_100-2_FINN_LE_BLANC", "Boyer_100-3_TAVERNE_DE_GALWAY",
                  "Buleon_047-6_SISYPHE", "Buleon_049-7_LE_LIVRE_DE_JOB", "Calandry_044-8_CONTE_DE_LA_CREATION",
                  "Calandry_047-7_HEPHAISTOS", "Calandry_061-2_MANGE_MA_GRAISSE", "Caudal_LANKOU_4", "Caudal_LANKOU_6",
                  "Cevin_060-1_LE_FILS_DU_ROI_ET_LE_CORBEAU", "Cevin_060-3_LE_MARCHAND",
                  "De_La_Salle_GRAND-MERE_MENSONGE_3", "De_La_Salle_GRAND-MERE_MENSONGE_4", "Desnouveaux_017-5_ZAHARA",
                  "Desnouveaux_076-4_HISTOIRE_DE_FEMMES", "Guillemin_037-6_HISTOIRE_DU_ROI_FREUDI",
                  "Guillemin_038-5_AMOUR_ET_PSYCHE", "Guillemin_042-2_PHAETON", "Guillemin_044-3_LE_ROI_LYCAON",
                  "Guillemin_056-10_LES_BABOUCHES", "Kiss_202i-2_LHERITIER_DE_LEMPEREUR",
                  "Kiss_202i-3_NASREDINE-LE_FOU_SAGE_ET_LA_VIEILLESSE", "Kiss_202i-4_NASREDINE_ET_LHOMME_DESESPERE",
                  "Kiss_202i-5_LE_VOYAGEUR_ET_LOISEAU_RARE", "Kiss_202i-7-8_LE_ROI_ET_LE_MISERABLE",
                  "Kiss_202i-8-9_LES_40_AFRICAINS_ET_LE_GENIE", "Kiss_202i-9-10_LE_PAYSAN_ET_LA_PAYSANNE",
                  "Kiss_202i-10-11_LES_BAIES_DAMOUR", "Kiss_202i-11-12_TITETE_ET_TICORPS",
                  "Mastre_106-1_CONTE_DE_BOURGOGNE_1", "Nataf_041-2_LA_REINE_JUMENT", "Nataf_041-3_LE_ROI_DES_CORBEAUX",
                  "Quere_063-2_1001_NUITS_2", "Sauvage_044-11_GENESE", "Sauvage_069-6_SHAMSEDINE_1",
                  "Sauvage_071-4_IRAGNAKA_1", "Sauvage_071-9_LE_CORBEAU-LE_RAT-LA_TORTUE_ET_LE_DAIM",
                  "Sauvage_080-5_SHAMSEDINE_2"]
    dev_list = ['Bizouerne_062-9_BARBIER_FIN', 'Bloch_052-5_DEUX_ECOSSAIS', 'Buleon_049-7_LE_LIVRE_DE_JOB',
                'Calandry_061-2_MANGE_MA_GRAISSE', 'Caudal_LANKOU_6', 'Desnouveaux_076-4_HISTOIRE_DE_FEMMES',
                'Guillemin_056-10_LES_BABOUCHES', 'Kiss_202i-11-12_TITETE_ET_TICORPS',
                'Mastre_106-1_CONTE_DE_BOURGOGNE_1', 'Nataf_041-2_LA_REINE_JUMENT', 'Quere_063-2_1001_NUITS_2',
                'Sauvage_044-11_GENESE', 'Sauvage_080-5_SHAMSEDINE_2']
    is_test = file_name in test_list
    is_train = file_name in train_list
    is_dev = file_name in dev_list
    return is_test, is_dev, is_train


def split_file_synpaflex(file, file_name):
    file_test, file_train = "", ""
    file_lines = file.split("\n")
    if file_name == "chevalier_filleDuPirate.txt":
        file_train = file_lines[:293]
        file_test = file_lines[293:]
    elif file_name == "feval_vampire.txt":
        file_train = file_lines[:16]
        file_test = file_lines[16:]
    elif file_name == "flaubert_MmeBovary.txt":
        file_train = file_lines[:36]
        file_test = file_lines[36:]
    elif file_name == "merime_carmen.txt":
        file_train = file_lines[:118]
        file_test = file_lines[118:]
    elif file_name == "merime_venusdille.txt":
        file_train = file_lines[:63]
        file_test = file_lines[63:]
    elif file_name == "sue_mysteresdeParis.txt":
        file_train = file_lines[:971]
        file_test = file_lines[971:]
    elif file_name == "zelter_contes.txt":
        file_train = file_lines[:44]
        file_test = file_lines[44:]
    return "\n".join(file_test), "\n".join(file_train)


def main(input_path, output_path, data_type):
    os.makedirs(output_path + "test", exist_ok=True)
    os.makedirs(output_path + "dev", exist_ok=True)
    os.makedirs(output_path + "train", exist_ok=True)
    for file_name in os.listdir(input_path):
        file = open(input_path + "" + file_name).read()

        if data_type == "orfeo":
            is_test, is_dev, is_train = file_status_orfeo(file_name)
            df_formated = format_file(file, data_type)
            if is_test:
                df_formated.to_csv(
                    output_path + "test/" + file_name.replace("txt", "csv").replace("orfeo", "csv"), index=False)
            elif is_train:
                df_formated.to_csv(
                    output_path + "train/" + file_name.replace("txt", "csv").replace("orfeo", "csv"), index=False)
                if is_dev:
                    df_formated.to_csv(
                        output_path + "dev/" + file_name.replace("txt", "csv").replace("orfeo", "csv"), index=False)

        elif data_type == "synpaflex":
            file_test, file_train = split_file_synpaflex(file, file_name)

            df_formated_test = format_file(file_test, data_type)
            df_formated_train = format_file(file_train, data_type)

            df_formated_test.to_csv(
                output_path + "test/" + file_name.replace("txt", "csv").replace("orfeo", "csv"), index=False)
            df_formated_train.to_csv(
                output_path + "train/" + file_name.replace("txt", "csv").replace("orfeo", "csv"), index=False)
            if file_name in ["chevalier_filleDuPirate.txt", "feval_vampire.txt", "flaubert_MmeBovary.txt"] :
                df_formated_train.to_csv(
                    output_path + "dev/" + file_name.replace("txt", "csv").replace("orfeo", "csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation of the break detection system")
    parser.add_argument("--input", help="path of the directory to be formatted",
                        default="data/init/corpus_cefc-orfeo/all_corpus/")
    parser.add_argument("--output", help="path of directory where the output of the script should be stored",
                        default="data/tmp/")
    parser.add_argument("--data_type", help="type of input data (orfeo or synpaflex)", default="orfeo")
    args = parser.parse_args()
    main(args.input, args.output, args.data_type)
