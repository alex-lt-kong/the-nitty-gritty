# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: calculator.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='calculator.proto',
  package='mytest',
  syntax='proto3',
  serialized_options=b'\n\017io.grpc.test.myB\013MyTestProtoP\001\242\002\003HLW',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x10\x63\x61lculator.proto\x12\x06mytest\"/\n\x11\x43\x61lculatorRequest\x12\x0c\n\x04num1\x18\x01 \x01(\x01\x12\x0c\n\x04num2\x18\x02 \x01(\x01\"!\n\x0f\x43\x61lculatorReply\x12\x0e\n\x06result\x18\x01 \x01(\x01\x32\x88\x01\n\nCalculator\x12;\n\x03\x61\x64\x64\x12\x19.mytest.CalculatorRequest\x1a\x17.mytest.CalculatorReply\"\x00\x12=\n\x05minus\x12\x19.mytest.CalculatorRequest\x1a\x17.mytest.CalculatorReply\"\x00\x42&\n\x0fio.grpc.test.myB\x0bMyTestProtoP\x01\xa2\x02\x03HLWb\x06proto3'
)




_CALCULATORREQUEST = _descriptor.Descriptor(
  name='CalculatorRequest',
  full_name='mytest.CalculatorRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='num1', full_name='mytest.CalculatorRequest.num1', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='num2', full_name='mytest.CalculatorRequest.num2', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28,
  serialized_end=75,
)


_CALCULATORREPLY = _descriptor.Descriptor(
  name='CalculatorReply',
  full_name='mytest.CalculatorReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='result', full_name='mytest.CalculatorReply.result', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=77,
  serialized_end=110,
)

DESCRIPTOR.message_types_by_name['CalculatorRequest'] = _CALCULATORREQUEST
DESCRIPTOR.message_types_by_name['CalculatorReply'] = _CALCULATORREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

CalculatorRequest = _reflection.GeneratedProtocolMessageType('CalculatorRequest', (_message.Message,), {
  'DESCRIPTOR' : _CALCULATORREQUEST,
  '__module__' : 'calculator_pb2'
  # @@protoc_insertion_point(class_scope:mytest.CalculatorRequest)
  })
_sym_db.RegisterMessage(CalculatorRequest)

CalculatorReply = _reflection.GeneratedProtocolMessageType('CalculatorReply', (_message.Message,), {
  'DESCRIPTOR' : _CALCULATORREPLY,
  '__module__' : 'calculator_pb2'
  # @@protoc_insertion_point(class_scope:mytest.CalculatorReply)
  })
_sym_db.RegisterMessage(CalculatorReply)


DESCRIPTOR._options = None

_CALCULATOR = _descriptor.ServiceDescriptor(
  name='Calculator',
  full_name='mytest.Calculator',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=113,
  serialized_end=249,
  methods=[
  _descriptor.MethodDescriptor(
    name='add',
    full_name='mytest.Calculator.add',
    index=0,
    containing_service=None,
    input_type=_CALCULATORREQUEST,
    output_type=_CALCULATORREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='minus',
    full_name='mytest.Calculator.minus',
    index=1,
    containing_service=None,
    input_type=_CALCULATORREQUEST,
    output_type=_CALCULATORREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_CALCULATOR)

DESCRIPTOR.services_by_name['Calculator'] = _CALCULATOR

# @@protoc_insertion_point(module_scope)
