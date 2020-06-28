from sklearn.metrics import roc_auc_score
import tensorflow as tf

class Metrics:
    def my_roc_auc_score(self,y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return 0.5

    def auroc(self,y_true, y_pred):
        try:
            return tf.py_func(self.my_roc_auc_score, (y_true, y_pred), tf.double)
        except:
            return 0
