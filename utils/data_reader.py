from collections import defaultdict, Counter
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))

def read_data(path_1, path_2, path_3):
    with open(path_1, 'r', encoding='utf-8') as f1, \
            open(path_2, 'r', encoding='utf-8') as f2, \
            open(path_3, 'r', encoding='utf-8') as f3:
        words = []
        # print(f1)
        for line in f1:
            words = line.split()

        for line in f2:
            words += line.split(' ')

        for line in f3:
            words += line.split(' ')

    return words


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    vocab, reverse_vocab = [], []
    print(f"词共计 {len(items)} ")
    dic = Counter(items)
    print(f"词汇共计 {len(dic)}")

    if sort:
        # sort by count
        # 按照字典里的词频进行排序，出现次数多的排在前面
        dic_items = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    else:
        dic_items = sorted(dic.items(), key=lambda x: x[0], reverse=True)

    for index, item in enumerate(dic_items):
        if min_count and min_count > item[1]:
            continue
        vocab.append((item[0], index))
        reverse_vocab.append((index, item[1]))


    """
    建立项目的vocab和reverse_vocab，vocab的结构是（词，index）
    vocab = (one line)
    reverse_vocab = (one line)
    """

    return vocab, reverse_vocab


if __name__ == '__main__':
    lines = read_data('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
                      '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
                      '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR))
    vocab, reverse_vocab = build_vocab(lines)
    save_word_dict(vocab, '{}/vocab.txt'.format(BASE_DIR))
    save_word_dict(reverse_vocab, '{}/reverse_vocab.txt'.format(BASE_DIR))