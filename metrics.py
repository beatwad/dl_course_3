def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp = len([i for i in range(len(prediction)) if (prediction[i] == ground_truth[i]) & (prediction[i])])
    fp = len([i for i in range(len(prediction)) if (prediction[i] != ground_truth[i]) & (prediction[i])])
    tn = len([i for i in range(len(prediction)) if (prediction[i] == ground_truth[i]) & (not prediction[i])])
    fn = len([i for i in range(len(prediction)) if (prediction[i] != ground_truth[i]) & (not prediction[i])])
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    accuracy = (tp + tn)/len(prediction)
    f1 = 2/(1/precision + 1/recall)
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    correct = len([i for i in range(len(prediction)) if (prediction[i] == ground_truth[i])])
    accuracy = correct/len(prediction)
    return accuracy
