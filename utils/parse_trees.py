import os
import nltk
import spacy
import pickle
import benepar
import pandas as pd
from nltk.tree import *


# global variables - bc it's easier this way
nlp = spacy.load('en_core_web_sm')
benepar.download('benepar_en3')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
spacy.prefer_gpu()


# couldn't be bothered with figuring out the package import stuff
# so just copied the two read file utils from utils
# and that's that...
def read_csv_file(file_path, sep=";"):
    df = pd.read_csv(file_path, sep=sep)

    return df


def read_json_file(file_path):
    json_obj = pd.read_json(path_or_buf=file_path, lines=True)

    return json_obj


def delete_leaves(tree):
    """
    tree is immutable, no return
    recursively removes the leave nodes from a parse tree

    :param tree: nltk tree
    """
    for subtree in tree:
        # if a subtree has leaves and its first child is not a Tree
        # then traversal has reached the leaf - go ahead and delete the leaf child
        if subtree.leaves() and type(subtree[0]) is not nltk.tree.Tree:
            del subtree[0]

        if type(subtree) is nltk.tree.Tree:
            delete_leaves(subtree)


def prune_depth(tree, depth=3):
    """
    removes nodes from a certain depth down of a parse tree
    using this so we can stop going too deep in a parse
    will always prune the leaves

    :param tree: nltk tree object
    :param depth: how many levels down to prune
    """
    n_leaves = tree.leaves()
    leavepos = {leave: tree.leaf_treeposition(index) for index, leave in enumerate(n_leaves)}

    for word, index in leavepos.items():
        if len(index) > depth:
            # since we can delete a whole swath of subtrees at once
            # if a subtree has multiple subnodes, this will throw an index error
            # we catch that and ignore it
            try:
                del tree[(index[0:depth - 1], 0)]
            except IndexError:
                pass

    delete_leaves(tree)


def generate_parse(text: str, depth=3):
    """
    :param text: just a string text - to be parsed
    :param depth: how many children down before we prune the tree
    """
    trees = []
    try:
        doc = nlp(text)
        for sent in list(doc.sents):
            tree = Tree.fromstring(sent._.parse_string)
            prune_depth(tree, depth)
            # for some reason - there are "\n" tokens in the parse string
            # when it doesn't really mean anything - we're just removing them
            tree = str(tree).replace("\n", "")
            trees.append(tree)
    except AssertionError:
        pass

    return trees


def generate_freq_category(parse_dict: dict):
    """
    categorises each parse into a category from 1 - 11
    with category 1 being top 1%, 2 being top 10%, 3 is top 20%, etc.

    methodology taken from this paper:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5644354/
    :param parse_dict: key = parse string, value = counts
    :return:
    """
    # ensure parse dictionary is sorted
    sorted_parse = {
        k.replace("\n ", ""): v for k, v in sorted(
            parse_dict.items(), key=lambda item: item[1], reverse=True
        )
    }
    parse_list = list(sorted_parse)

    parse_category_dict = {}
    for index, parse in enumerate(parse_list):
        if index < .01 * len(parse_list):
            parse_category_dict.update({parse: 1})
        elif index < .1 * len(parse_list):
            parse_category_dict.update({parse: 2})
        elif index < .2 * len(parse_list):
            parse_category_dict.update({parse: 3})
        elif index < .3 * len(parse_list):
            parse_category_dict.update({parse: 4})
        elif index < .4 * len(parse_list):
            parse_category_dict.update({parse: 5})
        elif index < .5 * len(parse_list):
            parse_category_dict.update({parse: 6})
        elif index < .6 * len(parse_list):
            parse_category_dict.update({parse: 7})
        elif index < .7 * len(parse_list):
            parse_category_dict.update({parse: 8})
        elif index < .8 * len(parse_list):
            parse_category_dict.update({parse: 9})
        elif index < .9 * len(parse_list):
            parse_category_dict.update({parse: 10})
        elif index < len(parse_list):
            parse_category_dict.update({parse: 11})

    return parse_category_dict


def generate_parse_trees_distrib(texts: list, depth=3, debug=True, pkl_file=None):
    """
    creates a distribution of types of parse trees
    :return: dictionary of parses to counts
    """
    parse_dict = {}
    num_fails = 0
    for text in texts:
        try:
            trees = generate_parse(text, depth)
            for tree in trees:
                # we keep count of each type of parse tree and increase it by one
                parse_dict.update({
                    str(tree): parse_dict.get(str(tree)) + 1 if parse_dict.get(str(tree)) else 1
                })
        # I'm not sure why sometimes the nlp call fails - maybe it just fails because
        # some weirdness in the Twitter dataset.
        except AssertionError:
            num_fails += 1

    if debug:
        print(f"failed to parse: {num_fails} sentences")

    if pkl_file:
        with open(pkl_file, 'wb+') as file:
            pickle.dump(parse_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    return parse_dict


def tweepfake():
    train = read_csv_file("../data/tweepfake/train.csv")
    train = train[train.text.str.len() < 512]
    human_tweets = list(train[train["account.type"] == "human"]["text"])
    bot_tweets = list(train[train["account.type"] == "bot"]["text"])

    generate_parse_trees_distrib(
        human_tweets,
        debug=True,
        pkl_file="../data/tweepfake/human_tweet_parse_count.pkl"
    )

    generate_parse_trees_distrib(
        bot_tweets,
        debug=True,
        pkl_file="../data/tweepfake/bot_tweet_parse_count.pkl"
    )


def abstract_cheat():
    human_abstract = read_json_file("../data/cheat/ieee-init.jsonl")
    bot_abstract = read_json_file("../data/cheat/ieee-chatgpt-generation.jsonl")

    human_abstract = human_abstract["abstract"]
    bot_abstract = bot_abstract["abstract"]

    generate_parse_trees_distrib(
        human_abstract,
        debug=True,
        pkl_file="../data/cheat/human_abstract_parse_count.pkl"
    )

    generate_parse_trees_distrib(
        bot_abstract,
        debug=True,
        pkl_file="../data/cheat/bot_abstract_parse_count.pkl"
    )


if __name__ == "__main__":
    tweepfake()
    abstract_cheat()
