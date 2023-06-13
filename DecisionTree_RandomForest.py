import numpy as np
import numpy.ma as ma
import math
from collections import Counter
import time


class DecisionNode:
    """Class to represent a nodes or leaves in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """
        Create a decision node with eval function to select between left and right node
        NOTE In this representation 'True' values for a decision take us to the left.
        This is arbitrary, but testing relies on this implementation.
        Args:
            left (DecisionNode): left child node
            right (DecisionNode): right child node
            decision_function (func): evaluation function to decide left or right
            class_label (value): label for leaf node
        """
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Determine recursively the class of an input array by testing a value
           against a feature's attributes values based on the decision function.

        Args:
            feature: (numpy array(value)): input vector for sample.

        Returns:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice index for data labels.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data contained in the ReadMe.
    It must be built fully starting from the root.
    
    Returns:
        The root node of the decision tree.
    """
    
    #Define Nodes
    dt_root = DecisionNode(None, None, lambda feature : feature[0] <= 0, None)
    dt_a2 = DecisionNode(None, None, lambda feature : feature[2] <= -0.7, None)
    dt_a3 = DecisionNode(None, None, lambda feature : feature[3] <= -0.5, None)
    
    
    #root node (A0 < 0) Left: y = 0, Right: Lead to A2 Decision Node   
    dt_root.left = DecisionNode(None, None, None, 0)
    dt_root.right = dt_a2
    
    #extend tree to A2 Node
    dt_a2.left = DecisionNode(None, None, None, 2)
    dt_a2.right = dt_a3
    
    #extend tree to A3
    dt_a3.left  = DecisionNode(None, None, None, 0)
    dt_a3.right = DecisionNode(None, None, None, 1)

   
    return dt_root


def confusion_matrix(true_labels, classifier_output, n_classes=2):
    """Create a confusion matrix to measure classifier performance.
   
    Classifier output vs true labels, which is equal to:
    Predicted  vs  Actual Values.
    
    Output will sum multiclass performance in the example format:
    (Assume the labels are 0,1,2,...n)
                                     |Predicted|
                     
    |A|            0,            1,           2,       .....,      n
    |c|   0:  [[count(0,0),  count(0,1),  count(0,2),  .....,  count(0,n)],
    |t|   1:   [count(1,0),  count(1,1),  count(1,2),  .....,  count(1,n)],
    |u|   2:   [count(2,0),  count(2,1),  count(2,2),  .....,  count(2,n)],'
    |a|   .............,
    |l|   n:   [count(n,0),  count(n,1),  count(n,2),  .....,  count(n,n)]]
    
    'count' function is expressed as 'count(actual label, predicted label)'.
    
    For example, count (0,1) represents the total number of actual label 0 and the predicted label 1;
                 count (3,2) represents the total number of actual label 3 and the predicted label 2.           
    
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
    Returns:
        A two dimensional array representing the confusion matrix.
    """
    c_matrix = np.ones((n_classes, n_classes))
    
    for col in range(c_matrix.shape[1]):
        for row in range(c_matrix.shape[0]):
            count = 0 
            i = 0 
            while i < len(true_labels):
                if true_labels[i] == row and classifier_output[i] == col:
                    count = count + 1
                i = i +1

            c_matrix[row][col] = count
    
    
    return c_matrix


def precision(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the precision of a classifier compared to the correct values.
    In this assignment, precision for label n can be calculated by the formula:
        precision (n) = number of correctly classified label n / number of all predicted label n 
                      = count (n,n) / (count(0, n) + count(1,n) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of precision of each classifier output. 
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [precision (0), precision(1), precision(2), ... precision(n)].
    """
    
    #If no confusion Matrix, build one
    if pe_matrix is None: 
        pe_matrix = np.ones((n_classes, n_classes))
    
        for col in range(pe_matrix.shape[1]):
            for row in range(pe_matrix.shape[0]):
                count = 0 
                i = 0 
                while i < len(true_labels):
                    if true_labels[i] == row and classifier_output[i] == col:
                        count = count + 1
                    i = i +1

                pe_matrix[row][col] = count
    
    #Calculate Precision = TP / TP + FP   
    
    precision = []
    
    for col in range(pe_matrix.shape[1]):
        false_counter = 0
        true_counter = 0
        for row in range(pe_matrix.shape[0]):
            if col == row:
                true_counter = pe_matrix[row][col]
            else:
                false_counter = false_counter + pe_matrix[row][col]
        
        precision.append((true_counter/(false_counter + true_counter)))
    
    return precision


def recall(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the recall of a classifier compared to the correct values.
    In this assignment, recall for label n can be calculated by the formula:
        recall (n) = number of correctly classified label n / number of all true label n 
                   = count (n,n) / (count(n, 0) + count(n,1) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of recall of each classifier output..
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [recall (0), recall (1), recall (2), ... recall (n)].
    """

    #If no confusion Matrix, build one
    if pe_matrix is None: 
        pe_matrix = np.ones((n_classes, n_classes))
    
        for col in range(pe_matrix.shape[1]):
            for row in range(pe_matrix.shape[0]):
                count = 0 
                i = 0 
                while i < len(true_labels):
                    if true_labels[i] == row and classifier_output[i] == col:
                        count = count + 1
                    i = i +1

                pe_matrix[row][col] = count
    
    #Calculate Recall = TP / TP + FN   
    
    recall = []
    
    for row in range(pe_matrix.shape[0]):
        false_counter = 0
        true_counter = 0
        for col in range(pe_matrix.shape[1]):
            if row == col:
                true_counter = pe_matrix[row][col]
            else:
                false_counter = false_counter + pe_matrix[row][col]
        
        recall.append((true_counter/(false_counter + true_counter)))
    
    return recall
            
    
    


def accuracy(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """Get the accuracy of a classifier compared to the correct values.
    Balanced Accuracy Weighted:
    -Balanced Accuracy: Sum of the ratios (accurate divided by sum of its row) divided by number of classes.
    -Balanced Accuracy Weighted: Balanced Accuracy with weighting added in the numerator and denominator.

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The accuracy of the classifier output.
    """
   
    
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    
    return np.sum(classifier_output == true_labels) / true_labels.shape[0]



def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0, 1, 2, ...
    Returns:
        Floating point number representing the gini impurity.
    """

    classes = list(set(class_vector))
    
    class_vector = np.array(class_vector)
    prob_vector = []
    
    for item in classes:
        prob_vector.append(np.sum(class_vector == item) / class_vector.shape[0])

    prob = 1
    for item in prob_vector:
        prob = prob - item**2   
        
    return prob


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0, 1, 2....
        current_classes (list(list(int): A list of lists where each list has
            0, 1, 2, ... values).
    Returns:
        Floating point number representing the gini gain.
    """

    previous_entropy = gini_impurity(previous_classes)
    
    previous_classes = np.array(previous_classes)
    
    remainder = 0
    for classes in current_classes:
        classes = np.array(classes)
        if classes.size == 0: continue
        remainder = remainder + gini_impurity(classes)*(classes.shape[0]/previous_classes.shape[0])
        
    return previous_entropy - remainder


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=22):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        
        rows =np.size(features,axis=0)
        col =np.size(features,axis=1)
        
        c=Counter(classes)
        if len(c)==1:
            return DecisionNode(None,None,None, class_label=classes[0])
        
        
        if not features.shape[0] or depth > self.depth_limit:
            
            return DecisionNode(None,None,None, class_label=max(c, key=c.get))
        
        #Split Point
        mean=np.mean(features,axis=0)
        
        
        left=[]
        right=[]
        max_gain=0
        max_gain_index=0
        
        for i in range(col):
            left_class = classes[features[:,i] < mean[i]]
            right_class = classes[features[:,i] >= mean[i]]
            if np.size(left_class)!=0 and np.size(right_class)!=0:
                gain = gini_gain(classes,[left_class.tolist(),right_class.tolist()])
            else:
                gain = 0 
            if max_gain<gain:
                max_gain=gain
                max_gain_index=i
        split_feature=features[:,max_gain_index]
        
        #Split on features
        left_features=features[split_feature < mean[max_gain_index]]
        right_features=features[split_feature >= mean[max_gain_index]]
        
        #Split on classes
        left_classes=classes[split_feature < mean[max_gain_index]]
        right_classes=classes[split_feature >= mean[max_gain_index]]
        
        #Check if class is empty
        if np.size(left_classes)==0 or np.size(right_classes)==0:
            if (c[0]>c[1]):
                return DecisionNode(None,None,None,0)
            else:
                return DecisionNode(None,None,None,1)
        
        #Add a layer    
        left=self.__build_tree__(left_features,left_classes,depth+1)
        right=self.__build_tree__(right_features,right_classes,depth+1)
        
        return DecisionNode(left,right,lambda features:features[max_gain_index] < mean[max_gain_index])

        
        
        

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = []
        rows = np.size(features,axis=0)
        
        
        for i in range(rows):
            class_labels.append(self.root.decide(features[i,:]))
            
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """
    folds = []
    features=dataset[0]
    num_examples=np.size(features,axis=0)
    test_size=int(np.floor(num_examples/k))
    sample_index=np.random.choice(num_examples,k*test_size,replace=False)
    k_folds=[]
    train_classes=dataset[1]
    train_features=dataset[0]
    
    #Iterate through k folds
    for i in range(k):
        #choose random index point 
        rand = sample_index[i*test_size:(i+1)*test_size]
        
        #seperate training data on index point
        this_feature = np.take(train_features,rand,axis=0)
        this_class = np.take(train_classes,rand)
        train_features = np.delete(train_features, rand, 0)
        train_classes = np.delete(train_classes, rand, 0)
        
        #set test and train set
        test = (this_feature,this_class)
        train = (train_features,train_classes)
        
        #append the data to folds
        folds.append((train,test))
        train_classes=dataset[1]
        train_features=dataset[0]
    return folds


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees=80, depth_limit=5, example_subsample_rate=.3,
                 attr_subsample_rate=.3):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """
        self.trees = []
        self.num_trees = 200
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.attr_index=[]

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        rows = np.size(features,axis=0)
        cols = np.size(features,axis=1)
        
        for i in range(self.num_trees):
            
            sub_example_index = np.random.choice(rows,int(rows * self.example_subsample_rate),replace=True)
            sub_example = np.take(features,sub_example_index,0)
            sub_example_class = np.take(classes,sub_example_index)
            sub_attr_index = np.random.choice(cols, int(cols * self.attr_subsample_rate),replace=False)
            sub_attr_example = np.take(sub_example,sub_attr_index,1)
            
            #Build decision tree
            tree = DecisionTree(self.depth_limit)
            
            tree.fit(sub_attr_example,sub_example_class)
            self.trees.append(tree)
            self.attr_index.append(sub_attr_index)

    def classify(self, features):
        votes = []
        rows = np.size(features,axis=0)
        
        for i in range(rows):
            vote=[]
            for j in range(len(self.trees)):
                index=self.attr_index[j]
                feature=features[i,:]
                sub_feature=np.take(feature,index)
                vote.append(self.trees[j].root.decide(sub_feature))

            c=Counter(vote)
            votes.append(c.most_common(1)[0][0])
        return votes


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, n_clf=0, depth_limit=0, example_subsample_rt=0.0, \
                 attr_subsample_rt=0.0, max_boost_cycles=0):
        """Create a boosting class which uses decision trees.
        Initialize and/or add whatever parameters you may need here.
        Args:
             num_clf (int): fixed number of classifiers.
             depth_limit (int): max depth limit of tree.
             attr_subsample_rate (float): percentage of attribute samples.
             example_subsample_rate (float): percentage of example samples.
        """
        self.num_clf = n_clf
        self.depth_limit = depth_limit
        self.example_subsample_rt = example_subsample_rt
        self.attr_subsample_rt=attr_subsample_rt
        self.max_boost_cycles = max_boost_cycles
        # TODO: finish this.
        raise NotImplemented()

    def fit(self, features, classes):
        """Build the boosting functions classifiers.
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        # TODO: finish this.
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features.
        Predict the labels for each feature in features to its corresponding class
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """
        # TODO: finish this.
        raise NotImplemented()


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """
        
       
        
        return (data ** 2) + data

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return (max_sum, max_sum_index)

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
            
        """
        data=data[0:100,:]
        sums=np.sum(data,axis=1)
        max_sum=np.amax(sums)
        max_sum_index=np.argmax(sums)
        return max_sum, max_sum_index
        
        
        
        

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        unique_dict = {}
        flattened = data.flatten()
        for item in flattened:
            if item > 0:
                if item in unique_dict:
                    unique_dict[item] += 1
                else:
                    unique_dict[item] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        
        a = np.unique(data[data > 0], return_counts=True)
        return zip(a[0], a[1])  
        

    def non_vectorized_glue(self, data, vector, dimension='c'):
        """Element wise array arithmetic with loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        if dimension == 'c' and len(vector) == data.shape[0]:
            non_vectorized = np.ones((data.shape[0],data.shape[1]+1), dtype=float)
            non_vectorized[:, -1] *= vector
        elif dimension == 'r' and len(vector) == data.shape[1]:
            non_vectorized = np.ones((data.shape[0]+1,data.shape[1]), dtype=float)
            non_vectorized[-1, :] *= vector
        else:
            raise ValueError('This parameter must be either c for column or r for row')
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row, col] = data[row, col]
        return non_vectorized

    def vectorized_glue(self, data, vector, dimension='c'):
        """Array arithmetic without loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        vectorized = None
        
        if dimension == 'c' and len(vector) == data.shape[0]:
            vectorized = np.append(data, [vector], axis=1)
            return vectorized
            
        elif dimension == 'r' and len(vector) == data.shape[1]:
            vectorized = np.append(data, [vector], axis=0)
            return vectorized
                    
        
        return vectorized

    def non_vectorized_mask(self, data, threshold):
        """Element wise array evaluation with loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared.
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        non_vectorized = np.zeros_like(data, dtype=float)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                val = data[row, col]
                if val >= threshold:
                    non_vectorized[row, col] = val
                    continue
                non_vectorized[row, col] = val**2

        return non_vectorized

    def vectorized_mask(self, data, threshold):
        """Array evaluation without loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared. You are required to use a binary mask for this problem
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        vectorized = ma.masked_less(data, threshold)
        
        
        
        
        return vectorized


def return_your_name():
    # return your name

    return 'Chase McGrail'
