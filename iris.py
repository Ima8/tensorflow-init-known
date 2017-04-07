import tensorflow as tf
import tensorflow.contrib.learn as learn
from sklearn import datasets, metrics

# available GPU memory  // per_process_gpu_memory_fraction=0.4 
# allow_growth=True   //  the GPU memory is not preallocated and will be able to grow as you need it
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

iris = datasets.load_iris()
feature_columns = learn.infer_real_valued_columns_from_input(iris.data)
classifier = learn.LinearClassifier(n_classes=3, feature_columns=feature_columns)
classifier.fit(iris.data, iris.target, steps=200, batch_size=32)
iris_predictions = list(classifier.predict(iris.data, as_iterable=True))
score = metrics.accuracy_score(iris.target, iris_predictions)
print("Accuracy: %f" % score)

