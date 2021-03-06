import tensorflow as tf 
import os 

SAVE_PATH = './save'
MODEL_NAME = 'test'
VERSION = 1
SERVE_PATH = './serve/{}/{}'.format(MODEL_NAME, VERSION)

checkpoint = tf.train.latest_checkpoint(SAVE_PATH)

tf.reset_default_graph()

with tf.Session() as sess:
    #import saved graph
    saver = tf.train.import_meta_graph(checkpoint + '.meta')
    graph = tf.get_default_graph()
    sess.run(tf.global_variables_initializer())
    inputs = graph.get_tensor_by_name('inputs:0')
    predictions = graph.get_tensor_by_name('prediction/Sigmoid:0')

    #build tensor info
    model_input = tf.saved_model.utils.build_tensor_info(inputs)
    model_output = tf.saved_model.utils.build_tensor_info(predictions)

    #create model signature (signature def)
    signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
        inputs = {'inputs': model_input},
        outputs = {'outputs': model_output},
        method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    #create model builder
    builder = tf.saved_model.builder.SavedModelBuilder(SERVE_PATH)

    builder.add_meta_graph_and_variables(
        sess, 
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_definition
        }
    )

    #save model builder so we can serve it with model server 
    builder.save()

