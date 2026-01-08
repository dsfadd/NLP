import numpy as np
from typing import Any
from sklearn.model_selection import train_test_split

def test_svm(vectors:list[Any],labels:list[str]):
    x = np.vstack(vectors)
    y = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    from sklearn.svm import SVC

    svm = SVC()
    svm.fit(x_train, y_train)

    from sklearn.metrics import classification_report

    y_pred = svm.predict(x_test)
    print(classification_report(y_test, y_pred))
