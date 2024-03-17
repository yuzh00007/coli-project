import nltk
import spacy
import pickle
import benepar
from nltk.tree import *

from utils.utils import read_csv_file


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


def generate_parse_trees(nlp, texts, debug=True, pkl_file=None):
    parse_dict = {}
    num_fails = 0
    for text in texts:
        try:
            doc = nlp(text)
            for sent in list(doc.sents):
                tree = Tree.fromstring(sent._.parse_string)
                prune_depth(tree, 3)
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
        with open(pkl_file, 'wb') as file:
            pickle.dump(parse_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    return parse_dict


def main():
    nlp = spacy.load('en_core_web_sm')
    benepar.download('benepar_en3')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
    spacy.prefer_gpu()

    train = read_csv_file("../data/tweepfake/train.csv")
    human_tweets = list(train[train["account.type"] == "human"]["text"])[:20]

    generate_parse_trees(nlp, human_tweets, pkl_file="./human_tweet_parse_count")


if __name__ == "__main__":
    ...
