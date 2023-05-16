# coding: utf-8
#!/usr/bin/python
import os

def SonoriPy(word, IPA=False):
    '''
    This program syllabifies words based on the Sonority Sequencing Principle (SSP)

    >>> SonoriPy("justification")
    ['jus', 'ti', 'fi', 'ca', 'tion']
    '''

    def no_syll_no_vowel(ss):
        '''
        cannot be a syllable without a vowel
        '''

        nss = []
        front = ""
        for i, syll in enumerate(ss):
            # if following syllable doesn't have vowel,
            # add it to the current one
            if not any(char in vowels for char in syll):
                if len(nss) == 0:
                    front += syll
                else:
                    nss = nss[:-1] + [nss[-1] + syll]
            else:
                if len(nss) == 0:
                    nss.append(front + syll)
                else:
                    nss.append(syll)

        return nss

    # SONORITY HIERARCHY, MODIFY FOR LANGUAGE BELOW
    # categories should be collapsed into more general groups
    vowels = 'aeiouyàáâäæãåāèéêëēėęîïíīįìôöòóœøōõûüùúūůÿ'
    approximates = ''
    nasals = 'lmnrw'
    fricatives = 'zvsf'
    affricates = ''
    stops = 'bcdgtkpqxhj'

    # SONORITY HIERARCHY for IPà
    if IPA:
        # categories can be collapsed into more general groups
        vowelcount = 0  # if vowel count is 1, syllable is automatically 1
        sylset = []  # to collect letters and corresponding values
        for letter in word.strip(".:;?!)('" + '"'):
            if letter.lower() in 'aɔʌã':
                sylset.append((letter, 9))
                vowelcount += 1  # to check for monosyllabic words
            elif letter.lower() in 'eéẽɛøoõ':
                sylset.append((letter, 8))
                vowelcount += 1  # to check for monosyllabic words
            elif letter.lower() in 'iu':
                sylset.append((letter, 7))
                vowelcount += 1  # to check for monosyllabic words
            elif letter.lower() in 'jwɥh':
                sylset.append((letter, 6))
            elif letter.lower() in 'rl':
                sylset.append((letter, 5))
            elif letter.lower() in 'mn':
                sylset.append((letter, 4))
            elif letter.lower() in 'zvðʒ':
                sylset.append((letter, 3))
            elif letter.lower() in 'sfθʃ':
                sylset.append((letter, 2))
            elif letter.lower() in 'bdg':
                sylset.append((letter, 1))
            elif letter.lower() in 'ptkx':
                sylset.append((letter, 0))
            else:
                sylset.append((letter, 0))

    # assign numerical values to phonemes (characters)
    vowelcount = 0  # if vowel count is 1, syllable is automatically 1
    sylset = []  # to collect letters and corresponding values
    for letter in word:
        if letter.lower() in vowels:
            sylset.append((letter, 5))
            vowelcount += 1
        elif letter.lower() in approximates:
            sylset.append((letter, 4))
        elif letter.lower() in nasals:
            sylset.append((letter, 3))
        elif letter.lower() in fricatives:
            sylset.append((letter, 2))
        elif letter.lower() in affricates:
            sylset.append((letter, 1))
        elif letter.lower() in stops:
            sylset.append((letter, 0))
        else:
            sylset.append((letter, 0))

    # SSP syllabification follows
    final_sylset = []
    if vowelcount == 1:  # finalize word immediately if monosyllabic
        final_sylset.append(word)
    if vowelcount != 1:
        syllable = ''  # prepare empty syllable to build upon
        for i, tup in enumerate(sylset):
            if i == 0:  # if it's the first letter, append automatically
                syllable += tup[0]

            else:
                # add whatever is left at end of word, last letter
                if i == len(sylset) - 1:
                    syllable += tup[0]
                    final_sylset.append(syllable)

                # MAIN ALGORITHM BELOW

                # these cases DO NOT trigger syllable breaks
                elif (i < len(sylset) - 1) and tup[1] < sylset[i + 1][1] and \
                        tup[1] > sylset[i - 1][1]:
                    syllable += tup[0]
                elif (i < len(sylset) - 1) and tup[1] > sylset[i + 1][1] and \
                        tup[1] < sylset[i - 1][1]:
                    syllable += tup[0]
                elif (i < len(sylset) - 1) and tup[1] > sylset[i + 1][1] and \
                        tup[1] > sylset[i - 1][1]:
                    syllable += tup[0]
                elif (i < len(sylset) - 1) and tup[1] > sylset[i + 1][1] and \
                        tup[1] == sylset[i - 1][1]:
                    syllable += tup[0]
                elif (i < len(sylset) - 1) and tup[1] == sylset[i + 1][1] and \
                        tup[1] > sylset[i - 1][1]:
                    syllable += tup[0]
                elif (i < len(sylset) - 1) and tup[1] < sylset[i + 1][1] and \
                        tup[1] == sylset[i - 1][1]:
                    syllable += tup[0]

                # these cases DO trigger syllable break
                # if phoneme value is equal to value of preceding AND following
                # phoneme
                elif (i < len(sylset) - 1) and tup[1] == sylset[i + 1][1] and \
                        tup[1] == sylset[i - 1][1]:
                    syllable += tup[0]
                    # append and break syllable BEFORE appending letter at
                    # index in new syllable
                    final_sylset.append(syllable)
                    syllable = ""

                # if phoneme value is less than preceding AND following value
                # (trough)
                elif (i < len(sylset) - 1) and tup[1] < sylset[i + 1][1] and \
                        tup[1] < sylset[i - 1][1]:
                    # append and break syllable BEFORE appending letter at
                    # index in new syllable
                    final_sylset.append(syllable)
                    syllable = ""
                    syllable += tup[0]

                # if phoneme value is less than preceding value AND equal to
                # following value
                elif (i < len(sylset) - 1) and tup[1] == sylset[i + 1][1] and \
                        tup[1] < sylset[i - 1][1]:
                    syllable += tup[0]
                    # append and break syllable BEFORE appending letter at
                    # index in new syllable
                    final_sylset.append(syllable)
                    syllable = ""

    final_sylset = no_syll_no_vowel(final_sylset)

    return (final_sylset)


def main(input, output, nb_syll):
    file_names = os.listdir(input)

    for i, file_name in enumerate(file_names, 1):
        # get ground truth labels
        with open(os.path.join(input, file_name), "r", encoding="utf8",) as f:
            lines = f.read().strip().split("\n")[1:]
            count_syll = 0
            output_txt = ""
            for word in lines:
                word = word.split(",")[0]
                word_syll = len(SonoriPy(word))
                count_syll += word_syll
                if count_syll < nb_syll:
                    output_txt += word + "\t0" + "\t0\n"
                else:
                    count_syll = 0
                    output_txt += word + "\t1" + "\t0\n"
            with open(output + file_name, 'w', encoding='utf-8') as of:
                of.write(output_txt)


if __name__ == "__main__":
    nb_syll = 15
    input = "data/orfeo-synpaflex/test"
    output = "results/orfeo-synpaflex_baseline_syll/" + str(nb_syll) + "_syll/"
    main(input, output, nb_syll)


