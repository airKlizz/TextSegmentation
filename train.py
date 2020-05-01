import tensorflow as tf
import numpy as np
from tqdm import tqdm
import argparse
from model.segmenter import Segmenter
from dataset.utils import create_tf_dataset
from metrics.confusion_matrix import ConfusionMatrix
from evaluation.utils import create_candidate, eval, print_metrics

@tf.function
def train_step(model, optimizer, loss, inputs, gold, mask, train_loss, train_acc, train_confusion_matrix):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss_value = loss(gold, predictions, mask)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss_value)
    train_acc(gold, predictions)
    train_confusion_matrix(gold, predictions)

@tf.function
def test_step(model, loss, inputs, gold, mask, validation_loss, validation_acc, validation_confusion_matrix):
    predictions = model(inputs, training=False)
    t_loss = loss(gold, predictions, mask)
    validation_loss(t_loss)
    validation_acc(gold, predictions)
    validation_confusion_matrix(gold, predictions)

def main(bidirectional, num_classification_layers, train_path, max_sentences, test_size, batch_size, epochs, learning_rate, epsilon, clipnorm, save_path, test_data_path, test_gold_path, candidate_path, evaluation_every_epoch):
    '''
    Load Hugging Face tokenizer and model
    '''
    model = Segmenter(max_sentences, bidirectional, num_classification_layers)

    '''
    Create train and validation dataset
    '''
    train_dataset, validation_dataset, train_length, validation_length = create_tf_dataset(train_path, max_sentences, test_size, batch_size)

    '''
    Initialize optimizer and loss function for training
    '''
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon, clipnorm=clipnorm)
    loss = tf.keras.losses.CategoricalCrossentropy()
    
    '''
    Define metrics
    '''
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    validation_acc = tf.keras.metrics.CategoricalAccuracy(name='validation_accuracy')
    train_confusion_matrix = ConfusionMatrix(2, name='train_confusion_matrix')
    validation_confusion_matrix = ConfusionMatrix(2, name='validation_confusion_matrix')

    '''
    Training loop over epochs
    '''
    model_save_path_step_template = save_path+'segmenter_epoch_{epoch:04d}_loss_{loss:.3f}.h5'
    template_epoch = '\nEpoch {}/{}: \nTrain Loss: {}, Acc: {}, Confusion matrix:\n{}\nValidation Loss: {}, Acc: {}, Confusion matrix:\n{}'
    previus_validation_loss = 10000000
    print('Evaluation every epoch: {}'.format(evaluation_every_epoch))

    for epoch in range(epochs):
        train_loss.reset_states() 
        validation_loss.reset_states()
        train_acc.reset_states()
        validation_acc.reset_states()
        train_confusion_matrix.reset_states()
        validation_confusion_matrix.reset_states()

        for inputs, gold, mask in tqdm(train_dataset, desc="Training in progress", total=int(train_length/batch_size+1)):
            train_step(model, optimizer, loss, inputs, gold, mask, train_loss, train_acc, train_confusion_matrix)

        for inputs, gold, mask in tqdm(validation_dataset, desc="Validation in progress", total=int(validation_length/batch_size+1)):
            test_step(model, loss, inputs, gold, mask, validation_loss, validation_acc, validation_confusion_matrix)

        print(template_epoch.format(epoch+1,
                                epochs,
                                train_loss.result(),
                                train_acc.result(),
                                train_confusion_matrix.result(),
                                validation_loss.result(),
                                validation_acc.result(),
                                validation_confusion_matrix.result()
                                ))
        if evaluation_every_epoch == True:
            create_candidate(model, test_data_path, candidate_path)
            metrics = eval(test_gold_path, candidate_path)
            print_metrics(metrics)
        
        if previus_validation_loss > validation_loss.result().numpy():
            previus_validation_loss = validation_loss.result().numpy()
            model_save_path_step = model_save_path_step_template.format(epoch=epoch, loss=previus_validation_loss)
            print('Saving: ', model_save_path_step)
            model.save_weights(model_save_path_step, save_format='h5')

        print('\n===========================================\n')
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''
    Variables for dataset
    '''
    parser.add_argument("--bidirectional", type=bool, help="True if you want a bidirectional RNN", default=True)
    parser.add_argument("--num_classification_layers", type=int, help="number classification layers", default=1)

    '''
    Variables for dataset
    '''
    parser.add_argument("--train_path", type=str, help="path to the train file", default="data/train.txt")
    parser.add_argument("--max_sentences", type=int, help="Number max of sentences in the text", default=32)
    parser.add_argument("--test_size", type=float, help="ratio of the test dataset", default=0.2)
    parser.add_argument("--batch_size", type=int, help="batch size", default=12)
    
    '''
    Variables for training
    '''
    parser.add_argument("--epochs", type=int, help="number of epochs", default=5)
    parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.001)
    parser.add_argument("--epsilon", type=float, help="epsilon", default=1e-8)
    parser.add_argument("--clipnorm", type=float, help="clipnorm", default=1.0)
    parser.add_argument("--save_path", type=str, help="path to the save folder", default="model/saved_weights/")

    '''
    Variables for evaluation
    '''
    parser.add_argument("--test_data_path", type=str, help="path to the test data file", default="data/test.data.txt")
    parser.add_argument("--test_gold_path", type=str, help="path to the test gold file", default="data/test.gold.txt")
    parser.add_argument("--candidate_path", type=str, help="path to the candidate file to save predictions", default="data/text.run.txt")
    parser.add_argument("--evaluation_every_epoch", type=bool, help="True if you want to evaluate at every epochs", default=False)
    
    '''
    Run main
    '''
    args = parser.parse_args()
    main(args.bidirectional, args.num_classification_layers, args.train_path, args.max_sentences, args.test_size, args.batch_size, args.epochs, args.learning_rate, args.epsilon, args.clipnorm, args.save_path, args.test_data_path, args.test_gold_path, args.candidate_path, args.evaluation_every_epoch)