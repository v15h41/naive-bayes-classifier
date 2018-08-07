import csv
import math
import random
from operator import itemgetter

CSV_FILE = ''

class Model:
    def __init__(self, prior_count, total_count, posterior_counts, attribute_counts):
        self.prior_count = prior_count
        self.total_count = total_count
        self.posterior_counts = posterior_counts
        self.attribute_counts = attribute_counts

def remove_class_column(data):
    new_data = []
    for row in data:
        new_data.append(row[:-1])

    return new_data

def normalize_row(row):
    #sum up the values        
    non_normalized_sum = 0
    for j in range(len(row)):
        non_normalized_sum += row[j]

    #normalize to make them equal to 1
    for j in range(len(row)):
        row[j] = row[j]/non_normalized_sum

    return row

def return_classes(data):
    class_column = len(data[0])-1
    classes = []
    #count number of classes in labelled data
    temp_class_dict = {}
    for row in data:
        if row[class_column] not in temp_class_dict:
            classes.append(row[class_column])
            temp_class_dict[row[class_column]] = True
    
    return classes

# This function should open a data file in csv, and transform it into a usable format 
def preprocess(name):
    data = []

    #read CSV into array
    with open(name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)

    return data

# This function should build a supervised NB model
def train_supervised(data):
    #find last column
    class_column = len(data[0])-1

    #store counts of all priors and a total count of instances
    prior_count = {}
    total_count = 0
    for row in data:
        if row[class_column] in prior_count:
            prior_count[row[class_column]] += 1
        else:
            prior_count[row[class_column]] = 1
        
        total_count += 1

    #a 1D array of dictionaries holding posterior values
    posterior_counts = [dict() for x in range(class_column)]
    #a 1D array holding the number of different attribute types per attribute
    attribute_counts = [0]*class_column
    
    #for each attribute  
    for i in range(class_column):
        #keep track of attribute types we've seen
        temp_attribute_dict = {}
        #for each instance
        for j in range(len(data)):
            #concatenate attribute and its corresponding instance class as a key
            attribute_class_key = data[j][i] + data[j][class_column]
            #store counts in dictionary
            if attribute_class_key in posterior_counts[i]:
                posterior_counts[i][attribute_class_key] += 1
            else:
                posterior_counts[i][attribute_class_key] = 1

            #count number of attribute values
            if data[j][i] not in temp_attribute_dict:
                attribute_counts[i] += 1
                temp_attribute_dict[data[j][i]] = True
        
    #stores all relevant values in a model class
    model = Model(prior_count, total_count, posterior_counts, attribute_counts)

    return model

# This function should predict the class for a set of instances, based on a trained model 
def predict_supervised(model, instances):
    predictions = []

    for instance in instances:
        prediction = []
        for prior_class in list(model.prior_count.keys()):
            #check if there are unseen events
            unseen = 0
            for i in range(len(instance)):
                attribute_class_key = instance[i] + prior_class
                if attribute_class_key not in model.posterior_counts[i]:
                    unseen = 1
                    break

            sum = math.log(model.prior_count[prior_class]/model.total_count)
            for i in range(len(instance)):
                attribute_class_key = instance[i] + prior_class
                if attribute_class_key in model.posterior_counts[i]:
                    #print (attribute_class_key, model.posterior_counts[i][attribute_class_key], "/", model.prior_count[prior_class], "op", model.attribute_counts[i])
                    sum += math.log((model.posterior_counts[i][attribute_class_key]+unseen)/(model.prior_count[prior_class]+unseen*model.attribute_counts[i]))
                else:
                    sum += math.log(1/(model.prior_count[prior_class]+model.attribute_counts[i]))
                
            class_tuple = (prior_class, sum)
            prediction.append(class_tuple)

        prediction.sort(key=itemgetter(1), reverse=True)
        predictions.append(prediction)      
            
    return predictions

# This function should evaluate a set of predictions, in a supervised context 
def evaluate_supervised(data, predictions):
    #find last column
    class_column = len(data[0])-1

    correct_predictions = 0
    for i in range(len(data)):
        if data[i][class_column] == predictions[i][0][0]:
            correct_predictions += 1

    return correct_predictions/len(data)

# This function should build an unsupervised NB model 
def train_unsupervised(data):
    #find last column
    class_column = len(data[0])-1

    num_classes = len(return_classes(data))
    class_probabilities = [[] for i in range(len(data))]

    #remove class column
    data = remove_class_column(data)

    for i in range(len(data)):
        class_probability_row = [0]*num_classes

        #assign random values to each class             
        for j in range(num_classes):
            class_probability_row[j] = random.random()

        #normalize and store in array
        class_probabilities[i] = normalize_row(class_probability_row)

    for i in range(50):
        prior_count = [0]*num_classes
        for class_probability_row in class_probabilities:
            for i in range(num_classes):
                prior_count[i] += class_probability_row[i]

        #a 1D array of dictionaries holding posterior values
        posterior_counts = [dict() for x in range(len(data[0]))]    

        for i in range(len(data[0])):
            for j in range(len(data)):
                if data[j][i] not in posterior_counts[i]:
                    posterior_counts[i][data[j][i]] = [0]*num_classes

                for k in range(num_classes):
                    posterior_counts[i][data[j][i]][k] += class_probabilities[j][k]

        for i in range(len(data)):
            for j in range(num_classes):
                if class_probabilities[i][j] > 0.9999999 or class_probabilities[i][j] < 0.00000001:
                    break

                sum = math.log(prior_count[j]/len(data))
                for k in range(len(data[0])):
                    #print(data[i][k], posterior_counts[k][data[i][k]][j],prior_count[j])                    
                    sum += math.log((posterior_counts[k][data[i][k]][j])/prior_count[j])

                class_probabilities[i][j] = math.exp(sum)

            class_probabilities[i] = normalize_row(class_probabilities[i])

    print(posterior_counts)

    model = Model(prior_count, len(data), posterior_counts, [])

    return model

# This function should predict the class distribution for a set of instances, based on a trained model
def predict_unsupervised(model, instances):
    predictions = []

    for instance in instances:
        prediction = [0]*len(model.prior_count)
        for i in range(len(model.prior_count)):
            sum = math.log(model.prior_count[i]/model.total_count)
            for k in range(len(instance)):
                sum += math.log((model.posterior_counts[k][instance[k]][i])/model.prior_count[i])
            prediction[i] = sum
        predictions.append(prediction)   
    
    return predictions

# This function should evaluate a set of predictions, in an unsupervised manner
def evaluate_unsupervised(data, predictions):
    #find last column
    class_column = len(data[0])-1
    classes = return_classes(data)

    confusion_matrix = [[0 for i in range(len(predictions[0]))] for j in range(len(predictions[0]))]

    

    for i, current_class in enumerate(classes):
        for j in range(len(classes)):
            for k in range(len(predictions)):                
                if data[k][class_column] == current_class and j == predictions[k].index(max(predictions[k])):
                    confusion_matrix[i][j] += 1
    
    maxes = [0]*len(classes)
    
    for j in range(len(classes)):
        current_max = 0
        for i in range(len(classes)):
            if current_max < confusion_matrix[i][j]:
                current_max = confusion_matrix[i][j]
        maxes[j] = current_max

    return sum(maxes)/len(predictions)

def evaluate_unsupervised_deterministic(data, predictions):
    #find last column
    class_column = len(data[0])-1
    classes = return_classes(data)

    correct = 0
    for i, row in enumerate(data):
        if row[class_column] == classes[predictions[i].index(max(predictions[i]))]:
            correct += 1

    return correct/len(predictions)

def main():
    data = preprocess(CSV_FILE)
    model = train_supervised(data)
    predictions = predict_supervised(model, remove_class_column(data))
    print(evaluate_supervised(data, predictions))

    model = train_unsupervised(data)
    predictions = predict_unsupervised(model, remove_class_column(data))
    print(evaluate_unsupervised(data, predictions))
    print(evaluate_unsupervised_deterministic(data, predictions))

main()
