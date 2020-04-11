from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

def get_labels(filename):
    with open(filename, 'r') as f:
        data = f.read()
        labels = []
        for article in data.split('\n\n'):
            if article == '' or article == '\n': continue
            for sentence in article.split('\n')[1:]:
                elems = sentence.split('\t')
                assert len(elems) == 3, 'File incorrect format : {}'.format(elems)
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
    print('gold labels: {}, candidate labels: {}'.format(len(gold_labels), len(candidate_labels)))
    return get_metrics(gold_labels, candidate_labels)

def create_candidate(model, test_data_input, output):
    
    with open(test_data_input, 'r') as f:
        with open(output, 'w') as output_f:
            
            data = f.read()
            articles = data.split('\n\n')
            for article in tqdm(articles, desc='Prediction in progress'):

                # Read article
                X_ = []
                text_ = []        
                sentences = article.split('\n')
                output_f.write(sentences[0]+'\n')    
                for sentence in sentences[1:]:
                    elems = sentence.split('\t')
                    if len(elems) != 3:
                        elems = elems[:2] + [" ".join(elems[2:])]
                    X_.append(elems[-1])
                    text_.append('\t'.join(elems[:2]))

                # Predict labels
                if len(X_) == 0: continue
                y_ = model.call(X_, prepare_inputs=True)
                y_ = tf.argmax(y_, axis=-1)
                y_ = list(tf.reshape(y_, (-1, )).numpy())
                assert len(y_) == len(X_), y_

                # Write
                for t, label in zip(text_, y_):
                    output_f.write(t+'\t'+str(label)+'\n')
                output_f.write('\n')

def print_metrics(metrics):
    print('=====================')
    print('Metrics:')
    print('---------------------')
    for m, v in metrics.items():
        print('{}:  \t{:.3f}'.format(m, v))
    print('=====================')