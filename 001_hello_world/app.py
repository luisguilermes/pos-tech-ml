# In this case, the training data consists of stocks information,
# and the labels indicates whether the stock price will go up(1) or down(0).
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Positive daily change
# High trading volume
# High closing price
stock01 = [1, 0, 1] # AAPL
stock02 = [0, 1, 0] # GOOGL
stock03 = [1, 1, 0] # MSFT
stock04 = [0, 0, 1] # AMZN
stock05 = [1, 1, 1] # TSLA
stock06 = [0, 1, 1] # FB

data_train = [stock01, stock02, stock03, stock04, stock05, stock06]
labels_train = [1, 0, 1, 0, 1, 0] # 1: price up, 0: price down

# Initialize the model LinearSVC
model = LinearSVC()
model.fit(data_train, labels_train)

# Set of test data
test1 = [1, 0, 0]
test2 = [0, 1, 1]
test3 = [1, 1, 0]

data_test = [test1, test2, test3]
labels_test = [1, 0, 1] # 1: price up, 0: price down

# Predict the labels for the test data
predictions = model.predict(data_test)
tax_accuracy = accuracy_score(labels_test, predictions)
print("Tax accuracy: %.2f%%" % (tax_accuracy * 100))