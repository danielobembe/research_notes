#ref: https://jacobbuckman.com/post/graph-inspection/

import tensorflow as tf 
a = tf.constant(2, name="a")
b = tf.constant(3, name='b')
c = a + b 

print("Our tf.Tensor objects: ")
print(a) 
print(b)
print(c)
print('\n')

a_op = a.op 
b_op = b.op 
c_op = c.op 

print("Our tf.Operation objects, printed in compressed form: ")
print(a_op.__repr__())
print(b_op.__repr__())
print(c_op.__repr__())
print('\n')

print("The default behavior of printing a tf.Operation object is to pretty-print")
print(c_op) 
print('\n')

print("Inpect the consumers of each tensor")
print(a.consumers())
print(b.consumers())
print(c.consumers())
print("\n")

print("Inspect input tensors for each op:")
# it's in a weird format, tensorflow.python.framework.ops._InputList, so we need to convert to list() to inspect
print(list(a_op.inputs))
print(list(b_op.inputs))
print(list(c_op.inputs))
print("\n")

print("Inspect output tensors for each op:")
# it's in a weird format, tensorflow.python.framework.ops._InputList, so we need to convert to list() to inspect
print(list(a_op.outputs))
print(list(b_op.outputs))
print(list(c_op.outputs))
print("\n")




print("The list of all nodes (TF.operations) in the graph: ")
g = tf.get_default_graph()
ops_list = g.get_operations()
print(ops_list)
print("\n")

print("The list of all edges (tf.Tensors) in the graph, by way of list comprehension:")
tensors_list = [tensor for op in ops_list for tensor in op.outputs]
print(tensors_list)
print("\n")