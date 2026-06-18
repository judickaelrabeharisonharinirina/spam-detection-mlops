from graphe import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
#division du dataset
X_train_text, X_test_text, y_train, y_test = train_test_split(x, y)
X_train = vector.fit_transform(X_train_text)
print(X_train.shape)

#perform LG
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
X_test = vector.transform(X_test_text)
#creation de prediction
y_pred =log_reg.predict(X_test)

#confusion matrix
conf = confusion_matrix(y_test,y_pred)

print(conf)

"""
#test 
nouveaux_messages = [
    "Hey friend, are we still meeting for coffee at 5 pm?",
    "URGENT! Your mobile number has won a £2000 prize! Call 09061701461 now to claim your cash award!" # Un faux spam
]
nouveaux_messages_numeriques = vector.transform(nouveaux_messages)
predictions = log_reg.predict(nouveaux_messages_numeriques)
print("\ntest")
for msg, pred in zip(nouveaux_messages, predictions):
    print(f"Message : '{msg}'")
    print(f"Résultat de l'IA : {pred}\n")
"""