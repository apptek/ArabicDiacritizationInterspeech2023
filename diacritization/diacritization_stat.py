# -*- coding: utf-8 -*-

# original code taken from the ALIOsm repository, MIT Licence
# https://github.com/AliOsm/arabic-text-diacritization

import argparse
import pickle as pkl
import os
import numpy

CONSTANTS_PATH = os.path.join(os.path.dirname(__file__), "constants")


def get_diacritic_class(idx, line, case_ending, arabic_letters, diacritic_classes):
    # Handle without case ending
    if not case_ending:
        end = True
        for i in range(idx + 1, len(line)):
            if line[i] not in diacritic_classes:
                end = line[i].isspace()
                break
        if end:
            return -1

    if idx + 1 >= len(line) or line[idx + 1] not in diacritic_classes:
        # No diacritic
        return 0

    diac = line[idx + 1]

    if idx + 2 >= len(line) or line[idx + 2] not in diacritic_classes:
        # Only one diacritic
        return diacritic_classes.index(diac) + 1

    diac += line[idx + 2]

    try:
        # Try the possibility of double diacritics
        return diacritic_classes.index(diac) + 1
    except Exception:
        try:
            # Try the possibility of reversed double diacritics
            return diacritic_classes.index(diac[::-1]) + 1
        except Exception:
            # Otherwise consider only the first diacritic
            return diacritic_classes.index(diac[0]) + 1


def get_diacritics_classes(line, case_ending, arabic_letters, diacritic_classes, style):
    classes = list()
    for idx, char in enumerate(line):
        if style == "Fadel":
            if char in arabic_letters:
                classes.append(get_diacritic_class(idx, line, case_ending, arabic_letters, diacritic_classes))
        elif style == "Zitouni":
            if char in diacritic_classes or char.isspace():
                continue
            classes.append(get_diacritic_class(idx, line, case_ending, arabic_letters, diacritic_classes))
    return classes


def clear_line(line, arabic_letters, diacritic_classes):
    line = " ".join(
        "".join([char if char in list(arabic_letters) + diacritic_classes + [" "] else " " for char in line]).split()
    )
    new_line = ""
    for idx, char in enumerate(line):
        if char not in diacritic_classes or (idx > 0 and line[idx - 1] != " "):
            new_line += char
    line = new_line
    new_line = ""
    for idx, char in enumerate(line):
        if char not in diacritic_classes or (idx > 0 and line[idx - 1] != " "):
            new_line += char
    return new_line


def calculate_der(
    original_file,
    target_file,
    arabic_letters,
    diacritic_classes,
    style,
    case_ending=True,
    no_diacritic=True,
    no_sukun=False,
):
    with open(original_file, "r") as file:
        original_content = file.readlines()

    with open(target_file, "r") as file:
        target_content = file.readlines()
    assert len(original_content) == len(target_content)

    equal = 0
    not_equal = 0
    for original_line, target_line in zip(original_content, target_content):
        if style == "Fadel":
            original_line = clear_line(original_line, arabic_letters, diacritic_classes)
            target_line = clear_line(target_line, arabic_letters, diacritic_classes)

        original_classes = get_diacritics_classes(original_line, case_ending, arabic_letters, diacritic_classes, style)
        target_classes = get_diacritics_classes(target_line, case_ending, arabic_letters, diacritic_classes, style)

        assert len(original_classes) == len(target_classes)

        for original_class, target_class in zip(original_classes, target_classes):
            if not no_diacritic and original_class == 0:
                continue
            if original_class == -1 and target_class != -1:
                print("WOW!")
            if original_class != -1 and target_class == -1:
                print("WOW!")
            if original_class == -1 and target_class == -1:
                continue

            if no_sukun:
                if original_class == 7:
                    original_class = 0
                if target_class == 7:
                    target_class = 0

            equal += original_class == target_class
            not_equal += original_class != target_class

    return round(not_equal / max(1, (equal + not_equal)) * 100, 2)


def calculate_confusion(
    original_file,
    target_file,
    arabic_letters,
    diacritic_classes,
    diacritic_names,
    style,
):
    wrong_matrix = numpy.zeros((len(diacritic_classes) + 1, len(diacritic_names) + 1))

    # class_lookup = {c: i for i, c in enumerate(["NONE"] + diacritic_classes)}
    name_lookup = {i: n for i, n in enumerate(["NONE"] + diacritic_names)}

    with open(original_file, "r") as file:
        original_content = file.readlines()

    with open(target_file, "r") as file:
        target_content = file.readlines()

    assert len(original_content) == len(target_content)

    total_count = 0
    for original_line, target_line in zip(original_content, target_content):
        if style == "Fadel":
            original_line = clear_line(original_line, arabic_letters, diacritic_classes)
            target_line = clear_line(target_line, arabic_letters, diacritic_classes)

        original_classes = get_diacritics_classes(original_line, True, arabic_letters, diacritic_classes, style)
        target_classes = get_diacritics_classes(target_line, True, arabic_letters, diacritic_classes, style)

        assert len(original_classes) == len(target_classes)

        for original_class, target_class in zip(original_classes, target_classes):
            total_count += 1
            if original_class != target_class:
                wrong_matrix[original_class][target_class] += 1

        confusion_entries = []
        indices_tuples = reversed(numpy.argsort(wrong_matrix, axis=None))
        for idx in indices_tuples:
            i, j = numpy.unravel_index(idx, wrong_matrix.shape)
            if i == j:
                continue
            confusion_entries.append(
                "%.2f%%: " % (wrong_matrix[i][j] / float(total_count) * 100) + name_lookup[i] + "=>" + name_lookup[j]
            )

    return confusion_entries


def calculate_wer(
    original_file,
    target_file,
    arabic_letters,
    diacritic_classes,
    style,
    case_ending=True,
    no_diacritic=True,
    no_sukun=False,
):
    with open(original_file, "r") as file:
        original_content = file.readlines()

    with open(target_file, "r") as file:
        target_content = file.readlines()

    assert len(original_content) == len(target_content)

    equal = 0
    not_equal = 0
    for idx, (original_line, target_line) in enumerate(zip(original_content, target_content)):
        if style == "Fadel":
            original_line = clear_line(original_line, arabic_letters, diacritic_classes)
            target_line = clear_line(target_line, arabic_letters, diacritic_classes)

        original_line = original_line.split()
        target_line = target_line.split()
        assert len(original_line) == len(target_line)

        for original_word, target_word in zip(original_line, target_line):
            original_classes = get_diacritics_classes(
                original_word, case_ending, arabic_letters, diacritic_classes, style
            )
            target_classes = get_diacritics_classes(target_word, case_ending, arabic_letters, diacritic_classes, style)

            assert len(original_classes) == len(target_classes)

            if len(original_classes) == 0:
                continue

            equal_classes = 0
            # has_sukun_confusion = False
            for original_class, target_class in zip(original_classes, target_classes):
                if not no_diacritic and original_class == 0:
                    equal_classes += 1
                    continue
                if no_sukun:
                    if original_class == 7:
                        original_class = 0
                    if target_class == 7:
                        target_class = 0
                equal_classes += original_class == target_class

            equal += equal_classes == len(original_classes)
            not_equal += equal_classes != len(original_classes)

    return round(not_equal / max(1, (equal + not_equal)) * 100, 2)


def calculate_ser(
    original_file,
    target_file,
    arabic_letters,
    diacritic_classes,
    style,
    case_ending=True,
    no_diacritic=True,
):
    with open(original_file, "r") as file:
        original_content = file.readlines()

    with open(target_file, "r") as file:
        target_content = file.readlines()

    assert len(original_content) == len(target_content)

    equal = 0
    not_equal = 0
    for idx, (original_line, target_line) in enumerate(zip(original_content, target_content)):
        if style == "Fadel":
            original_line = clear_line(original_line, arabic_letters, diacritic_classes)
            target_line = clear_line(target_line, arabic_letters, diacritic_classes)

        original_line = original_line.split()
        target_line = target_line.split()

        assert len(original_line) == len(target_line)

        equal_words = True
        for original_word, target_word in zip(original_line, target_line):
            original_classes = get_diacritics_classes(
                original_word, case_ending, arabic_letters, diacritic_classes, style
            )
            target_classes = get_diacritics_classes(target_word, case_ending, arabic_letters, diacritic_classes, style)

            assert len(original_classes) == len(target_classes)

            if len(original_classes) == 0:
                continue

            equal_classes = 0
            for original_class, target_class in zip(original_classes, target_classes):
                if not no_diacritic and original_class == 0:
                    equal_classes += 1
                    continue
                equal_classes += original_class == target_class

            if equal_classes != len(original_classes):
                equal_words = False

        equal += equal_words
        not_equal += not equal_words

    return round(not_equal / max(1, (equal + not_equal)) * 100, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate DER and WER")
    parser.add_argument("original_file_path", help="File path to original text")
    parser.add_argument("target_file_path", help="File path to target text")
    parser.add_argument(
        "-s",
        "--style",
        help="How to calculate DER and WER",
        required=False,
        default="Fadel",
        choices=["Zitouni", "Fadel"],
    )
    parser.add_argument(
        "--confusion",
        help="Calculate diacritics confusion and store to given file",
        default=None,
        type=str,
    )
    parser.add_argument("--write-values-to", help="write all values into a specified folder")
    args = parser.parse_args()

    with open(CONSTANTS_PATH + "/ARABIC_LETTERS_LIST.pickle", "rb") as file:
        ARABIC_LETTERS_LIST = pkl.load(file)

    with open(CONSTANTS_PATH + "/CLASSES_LIST.pickle", "rb") as file:
        CLASSES_LIST = pkl.load(file)

    with open(CONSTANTS_PATH + "/CLASSES_NAMES_LIST.pickle", "rb") as file:
        CLASSES_NAMES_LIST = pkl.load(file)

    der_including_with_case = calculate_der(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
    )
    der_including_without_case = calculate_der(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        case_ending=False,
    )
    der_excluding_with_case = calculate_der(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        no_diacritic=False,
    )
    der_excluding_without_case = calculate_der(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        case_ending=False,
        no_diacritic=False,
    )
    print("+---------------------------------------------------------------------------------------------+")
    print("|       |  With case ending  | Without case ending |  With case ending  | Without case ending |")
    print("|  DER  |------------------------------------------+------------------------------------------|")
    print("|       |          Including no diacritic          |          Excluding no diacritic          |")
    print("|-------+------------------------------------------+------------------------------------------|")
    print("|   %%   |        %5.2f       |        %5.2f        |        %5.2f       |        %5.2f        |"
        % (
            der_including_with_case,
            der_including_without_case,
            der_excluding_with_case,
            der_excluding_without_case,
        )
    )

    wer_including_with_case = calculate_wer(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
    )
    wer_including_without_case = calculate_wer(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        case_ending=False,
    )
    wer_exclude_with_case = calculate_wer(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        no_diacritic=False,
    )
    wer_exclude_without_case = calculate_wer(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        case_ending=False,
        no_diacritic=False,
    )
    print("+---------------------------------------------------------------------------------------------+")
    print("")
    print("+---------------------------------------------------------------------------------------------+")
    print("|       |  With case ending  | Without case ending |  With case ending  | Without case ending |")
    print("|  WER  |------------------------------------------+------------------------------------------|")
    print("|       |          Including no diacritic          |          Excluding no diacritic          |")
    print("|-------+------------------------------------------+------------------------------------------|")
    print("|   %%   |        %5.2f       |        %5.2f        |        %5.2f       |        %5.2f        |"
        % (
            wer_including_with_case,
            wer_including_without_case,
            wer_exclude_with_case,
            wer_exclude_without_case,
        )
    )

    ser_including_with_case = calculate_ser(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
    )
    ser_including_without_case = calculate_ser(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        case_ending=False,
    )
    ser_excluding_with_case = calculate_ser(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        no_diacritic=False,
    )
    ser_excluding_without_case = calculate_ser(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        case_ending=False,
        no_diacritic=False,
    )
    print("+---------------------------------------------------------------------------------------------+")
    print("")
    print("+---------------------------------------------------------------------------------------------+")
    print("|       |  With case ending  | Without case ending |  With case ending  | Without case ending |")
    print("|  SER  |------------------------------------------+------------------------------------------|")
    print("|       |          Including no diacritic          |          Excluding no diacritic          |")
    print("|-------+------------------------------------------+------------------------------------------|")
    print("|   %%   |        %5.2f       |        %5.2f        |        %5.2f       |        %5.2f        |"
        % (
            ser_including_with_case,
            ser_including_without_case,
            ser_excluding_with_case,
            ser_excluding_without_case,
        )
    )

    no_sukun_der_including_with_case = calculate_der(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        no_sukun=True,
    )
    no_sukun_der_including_without_case = calculate_der(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        case_ending=False,
        no_sukun=True,
    )
    no_sukun_der_excluding_with_case = calculate_der(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        no_diacritic=False,
        no_sukun=True,
    )
    no_sukun_der_excluding_without_case = calculate_der(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        case_ending=False,
        no_diacritic=False,
        no_sukun=True,
    )
    print("+---------------------------------------------------------------------------------------------+")
    print("")
    print("+---------------------------------------------------------------------------------------------+")
    print("+- NO SUKUN ----------------------------------------------------------------------------------+")
    print("+---------------------------------------------------------------------------------------------+")
    print("|       |  With case ending  | Without case ending |  With case ending  | Without case ending |")
    print("|  DER  |------------------------------------------+------------------------------------------|")
    print("|       |          Including no diacritic          |          Excluding no diacritic          |")
    print("|-------+------------------------------------------+------------------------------------------|")
    print("|   %%   |        %5.2f       |        %5.2f        |        %5.2f       |        %5.2f        |"
        % (
            no_sukun_der_including_with_case,
            no_sukun_der_including_without_case,
            no_sukun_der_excluding_with_case,
            no_sukun_der_excluding_without_case,
        )
    )

    no_sukun_wer_including_with_case = calculate_wer(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        no_sukun=True,
    )
    no_sukun_wer_including_without_case = calculate_wer(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        case_ending=False,
        no_sukun=True,
    )
    no_sukun_wer_excluding_with_case = calculate_wer(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        no_diacritic=False,
        no_sukun=True,
    )
    no_sukun_wer_excluding_without_case = calculate_wer(
        args.original_file_path,
        args.target_file_path,
        ARABIC_LETTERS_LIST,
        CLASSES_LIST,
        args.style,
        case_ending=False,
        no_diacritic=False,
        no_sukun=True,
    )
    print("+---------------------------------------------------------------------------------------------+")
    print("")
    print("+---------------------------------------------------------------------------------------------+")
    print("|       |  With case ending  | Without case ending |  With case ending  | Without case ending |")
    print("|  WER  |------------------------------------------+------------------------------------------|")
    print("|       |          Including no diacritic          |          Excluding no diacritic          |")
    print("|-------+------------------------------------------+------------------------------------------|")
    print("|   %%   |        %5.2f       |        %5.2f        |        %5.2f       |        %5.2f        |"
        % (
            no_sukun_wer_including_with_case,
            no_sukun_wer_including_without_case,
            no_sukun_wer_excluding_with_case,
            no_sukun_der_excluding_without_case,
        )
    )
    print("+---------------------------------------------------------------------------------------------+")

    if args.write_values_to:
        if not os.path.exists(args.write_values_to):
            os.mkdir(args.write_values_to)
        locals_copy = locals().copy()
        for key, var in locals_copy.items():
            if key.endswith("case"):
                with open(os.path.join(args.write_values_to, key), "wt") as f:
                    f.write(str(var))

    if args.confusion:
        confusion_entries = calculate_confusion(
            args.original_file_path,
            args.target_file_path,
            ARABIC_LETTERS_LIST,
            CLASSES_LIST,
            CLASSES_NAMES_LIST,
            args.style,
        )
        with open(args.confusion, "wt") as f:
            for e in confusion_entries:
                f.write(e + "\n")
