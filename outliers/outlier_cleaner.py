#!/usr/bin/python
from sklearn.svm.libsvm import predict


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    tuples = []
    for i in range(0,90):
        prediction = predictions[i][0]
        age = ages[i][0]
        net_worth = net_worths[i][0]
        error = prediction - net_worth
        tuple = age, net_worth, error
        tuples.append(tuple)

    sorted_by_error = sorted(tuples, key=lambda tup: tup[2])
    del sorted_by_error[-9:]

    cleaned_data = sorted_by_error

    ### your code goes here

    
    return cleaned_data

