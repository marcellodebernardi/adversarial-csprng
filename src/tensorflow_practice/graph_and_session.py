# TensorFlow uses dataflow graphs to model computation. This has several
# advantages, including parallelism, ease of distribution, portability
# and more.
#
# A tf.Graph object contains two kinds of relevant information:
#   1.  graph structure: nodes and edges of the graph, indicating how
#       individual operations are composed
#   2.  graph collections: you can associate a list of objects to a key
#       which is useful for example for referring to a layer in a
#       neural network collectively
