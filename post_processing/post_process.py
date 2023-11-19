import pandas as pd

from .address_correction import AddressCorrection
address_correction = AddressCorrection()

def correct(answer):
    if not answer:
        return ""

    answer_tokens = answer.split()
    answer = answer.lower()

    result, diff = address_correction.address_correction(answer, correct_th=40)
    result = result.replace(",", "")

    result_tokens = result.split()

    if not len(result_tokens) == len(answer_tokens):
        corrected_result = " ".join([x.capitalize() for x in result_tokens])
    else:
        for i in range(len(result_tokens)):
            # fix typo đắk lắk contain ' or not
            if  'ăk' in answer_tokens[i] and 'ắk' in result_tokens[i]:
                result_tokens[i] = result_tokens[i].replace('ắk', 'ăk')
            if  'ắk' in answer_tokens[i] and 'ăk' in result_tokens[i]:
                result_tokens[i] = result_tokens[i].replace('ăk', 'ắk')

            if 'quý' in answer_tokens[i].lower() and 'quí' in result_tokens[i].lower():
                result_tokens[i] = result_tokens[i].replace('quí', 'quý')
            if 'quí' in answer_tokens[i].lower() and 'quý' in result_tokens[i].lower():
                result_tokens[i] = result_tokens[i].replace('quý', 'quí')

            if answer_tokens[i][0].isupper():
                result_tokens[i] = result_tokens[i].capitalize()

        corrected_result = " ".join([x for x in result_tokens])

    return corrected_result.strip()
