from sklearn.metrics import precision_recall_fscore_support

def get_labels(filename):
    with open(filename, 'r') as f:
        data = f.read()
        labels = []
        for article in data.split('\n\n'):
            for sentence in article.split('\n')[1:]:
                elems = sentence.split('\t')
                assert len(elems) == 3, 'File incorrect format'
                labels.append(elems[-1])
    return labels

def get_metrics(gold_labels, candidate_labels):
    precision, recall, fscore, _ = precision_recall_fscore_support(gold_labels, candidate_labels, average='macro')
    return {
        'precision': precision,
        'recall': recall,
        'fscore': fscore
    }

def eval(gold_filename, candidate_filename):
    gold_labels = get_labels(gold_filename)
    candidate_labels = get_labels(candidate_filename)
    return get_metrics(gold_labels, candidate_labels)