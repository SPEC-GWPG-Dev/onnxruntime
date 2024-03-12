from onnx import helper, numpy_helper, TensorProto

import onnx
import numpy as np
import sys
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vocoder')

def clear_field(proto, field):
    proto.ClearField(field)
    return proto

def order_repeated_field(repeated_proto, key_name, order):
    order = list(order)
    repeated_proto.sort(key=lambda x: order.index(getattr(x, key_name)))

def make_node(op_type, inputs, outputs, name=None, doc_string=None, domain=None, **kwargs):
    node = helper.make_node(op_type, inputs, outputs, name, doc_string, domain, **kwargs)
    if doc_string == '':
        node.doc_string = ''
    order_repeated_field(node.attribute, 'name', kwargs.keys())
    return node

def make_graph(*args, doc_string=None, **kwargs):
    graph = helper.make_graph(*args, doc_string=doc_string, **kwargs)
    if doc_string == '':
        graph.doc_string = ''
    return graph

custom_node = make_node(
    'WithOptionalOutput',
    inputs=['GraphIn0'],
    outputs=['Output0', '', 'Output2'],
    name='CustomOptionalOutput',
    domain='test',
)

iden0 = make_node(
    'Identity',
    inputs=['Output0'],
    outputs=['GraphOut0'],
    name='Iden0',
)

iden1 = make_node(
    'Identity',
    inputs=['Output2'],
    outputs=['GraphOut1'],
    name='Iden2',
)

graph=make_graph(
    name='CustomOpTest',
    inputs=[helper.make_tensor_value_info('GraphIn0', TensorProto.FLOAT, shape=[2])],
    outputs=[helper.make_tensor_value_info('GraphOut0', TensorProto.FLOAT, shape=[2]), 
             helper.make_tensor_value_info('GraphOut1', TensorProto.FLOAT, shape=[2])],
    nodes=[custom_node, iden0, iden1]
)

model = helper.make_model(
    opset_imports=[clear_field(helper.make_operatorsetid('', 13), 'domain'), helper.make_operatorsetid('test', 1)],
    ir_version=7,
    graph=graph
)

if __name__ == '__main__':
    _, out_path = sys.argv
    onnx.save(model, 'custom_op_optional_output.onnx')
