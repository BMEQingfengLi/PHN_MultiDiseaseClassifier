def MulticlassACCScore(truelabel_list_np, predlabel_list_np):
    '''

    :param truelabel_list_np:
    :param predlabel_list_np:
    :return:
    '''
    correct_subj_counter = 0
    for idx in range(len(truelabel_list_np)):
        if truelabel_list_np[idx] == predlabel_list_np[idx]:
            correct_subj_counter += 1
    acc = float(correct_subj_counter) / len(truelabel_list_np)
    return acc