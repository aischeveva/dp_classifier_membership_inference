��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8��
|
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_34/kernel
u
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel* 
_output_shapes
:
��*
dtype0
s
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_34/bias
l
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes	
:�*
dtype0
{
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
* 
shared_namedense_35/kernel
t
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes
:	�
*
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/dense_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_34/kernel/m
�
*Adam/dense_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_34/bias/m
z
(Adam/dense_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*'
shared_nameAdam/dense_35/kernel/m
�
*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m*
_output_shapes
:	�
*
dtype0
�
Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_35/bias/m
y
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
_output_shapes
:
*
dtype0
�
Adam/dense_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_34/kernel/v
�
*Adam/dense_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_34/bias/v
z
(Adam/dense_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*'
shared_nameAdam/dense_35/kernel/v
�
*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v*
_output_shapes
:	�
*
dtype0
�
Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_35/bias/v
y
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
�%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�$
value�$B�$ B�$
�
layer_with_weights-0
layer-0
layer-1
	variables
regularization_losses
trainable_variables
	keras_api

signatures
�
layer-0
	layer_with_weights-0
	layer-1

layer_with_weights-1

layer-2
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api

0
1
2
3
 

0
1
2
3
�
	variables
non_trainable_variables
regularization_losses

layers
layer_metrics
trainable_variables
metrics
layer_regularization_losses
 
R
	variables
regularization_losses
trainable_variables
 	keras_api
h

kernel
bias
!	variables
"regularization_losses
#trainable_variables
$	keras_api
h

kernel
bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
�
)iter

*beta_1

+beta_2
	,decay
-learning_ratemRmSmTmUvVvWvXvY

0
1
2
3
 

0
1
2
3
�
	variables
.non_trainable_variables
regularization_losses

/layers
0layer_metrics
trainable_variables
1metrics
2layer_regularization_losses
 
 
 
�
	variables
3non_trainable_variables
regularization_losses

4layers
5layer_metrics
trainable_variables
6metrics
7layer_regularization_losses
KI
VARIABLE_VALUEdense_34/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_34/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_35/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_35/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
 
 
 
 
 
�
	variables
8non_trainable_variables
regularization_losses

9layers
:layer_metrics
trainable_variables
;metrics
<layer_regularization_losses

0
1
 

0
1
�
!	variables
=non_trainable_variables
"regularization_losses

>layers
?layer_metrics
#trainable_variables
@metrics
Alayer_regularization_losses

0
1
 

0
1
�
%	variables
Bnon_trainable_variables
&regularization_losses

Clayers
Dlayer_metrics
'trainable_variables
Emetrics
Flayer_regularization_losses
][
VARIABLE_VALUE	Adam/iter>layer_with_weights-0/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEAdam/beta_1@layer_with_weights-0/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEAdam/beta_2@layer_with_weights-0/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE
Adam/decay?layer_with_weights-0/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/learning_rateGlayer_with_weights-0/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
	1

2
 

G0
H1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Itotal
	Jcount
K	variables
L	keras_api
D
	Mtotal
	Ncount
O
_fn_kwargs
P	variables
Q	keras_api
db
VARIABLE_VALUEtotalIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEcountIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

I0
J1

K	variables
fd
VARIABLE_VALUEtotal_1Ilayer_with_weights-0/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEcount_1Ilayer_with_weights-0/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

M0
N1

P	variables
��
VARIABLE_VALUEAdam/dense_34/kernel/mWvariables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/dense_34/bias/mWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_35/kernel/mWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/dense_35/bias/mWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_34/kernel/vWvariables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/dense_34/bias/vWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_35/kernel/vWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/dense_35/bias/vWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
#serving_default_sequential_34_inputPlaceholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall#serving_default_sequential_34_inputdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_592536
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_34/kernel/m/Read/ReadVariableOp(Adam/dense_34/bias/m/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOp*Adam/dense_34/kernel/v/Read/ReadVariableOp(Adam/dense_34/bias/v/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_592820
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_34/kerneldense_34/biasdense_35/kerneldense_35/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_34/kernel/mAdam/dense_34/bias/mAdam/dense_35/kernel/mAdam/dense_35/bias/mAdam/dense_34/kernel/vAdam/dense_34/bias/vAdam/dense_35/kernel/vAdam/dense_35/bias/v*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_592893��
�	
�
D__inference_dense_35_layer_call_and_return_conditional_losses_592310

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
~
)__inference_dense_34_layer_call_fn_592715

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_5922842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_sequential_34_layer_call_and_return_conditional_losses_592629

inputs+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity��dense_34/BiasAdd/ReadVariableOp�dense_34/MatMul/ReadVariableOp�dense_35/BiasAdd/ReadVariableOp�dense_35/MatMul/ReadVariableOpu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
flatten_17/Const�
flatten_17/ReshapeReshapeinputsflatten_17/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_17/Reshape�
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_34/MatMul/ReadVariableOp�
dense_34/MatMulMatMulflatten_17/Reshape:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_34/MatMul�
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_34/BiasAdd/ReadVariableOp�
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_34/BiasAddt
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_34/Relu�
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02 
dense_35/MatMul/ReadVariableOp�
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_35/MatMul�
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_35/BiasAdd/ReadVariableOp�
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_35/BiasAdd�
IdentityIdentitydense_35/BiasAdd:output:0 ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_sequential_35_layer_call_and_return_conditional_losses_592576

inputs9
5sequential_34_dense_34_matmul_readvariableop_resource:
6sequential_34_dense_34_biasadd_readvariableop_resource9
5sequential_34_dense_35_matmul_readvariableop_resource:
6sequential_34_dense_35_biasadd_readvariableop_resource
identity��-sequential_34/dense_34/BiasAdd/ReadVariableOp�,sequential_34/dense_34/MatMul/ReadVariableOp�-sequential_34/dense_35/BiasAdd/ReadVariableOp�,sequential_34/dense_35/MatMul/ReadVariableOp�
sequential_34/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2 
sequential_34/flatten_17/Const�
 sequential_34/flatten_17/ReshapeReshapeinputs'sequential_34/flatten_17/Const:output:0*
T0*(
_output_shapes
:����������2"
 sequential_34/flatten_17/Reshape�
,sequential_34/dense_34/MatMul/ReadVariableOpReadVariableOp5sequential_34_dense_34_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,sequential_34/dense_34/MatMul/ReadVariableOp�
sequential_34/dense_34/MatMulMatMul)sequential_34/flatten_17/Reshape:output:04sequential_34/dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_34/dense_34/MatMul�
-sequential_34/dense_34/BiasAdd/ReadVariableOpReadVariableOp6sequential_34_dense_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_34/dense_34/BiasAdd/ReadVariableOp�
sequential_34/dense_34/BiasAddBiasAdd'sequential_34/dense_34/MatMul:product:05sequential_34/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
sequential_34/dense_34/BiasAdd�
sequential_34/dense_34/ReluRelu'sequential_34/dense_34/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_34/dense_34/Relu�
,sequential_34/dense_35/MatMul/ReadVariableOpReadVariableOp5sequential_34_dense_35_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02.
,sequential_34/dense_35/MatMul/ReadVariableOp�
sequential_34/dense_35/MatMulMatMul)sequential_34/dense_34/Relu:activations:04sequential_34/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
sequential_34/dense_35/MatMul�
-sequential_34/dense_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_34_dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_34/dense_35/BiasAdd/ReadVariableOp�
sequential_34/dense_35/BiasAddBiasAdd'sequential_34/dense_35/MatMul:product:05sequential_34/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2 
sequential_34/dense_35/BiasAdd�
softmax_17/SoftmaxSoftmax'sequential_34/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
softmax_17/Softmax�
IdentityIdentitysoftmax_17/Softmax:softmax:0.^sequential_34/dense_34/BiasAdd/ReadVariableOp-^sequential_34/dense_34/MatMul/ReadVariableOp.^sequential_34/dense_35/BiasAdd/ReadVariableOp-^sequential_34/dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2^
-sequential_34/dense_34/BiasAdd/ReadVariableOp-sequential_34/dense_34/BiasAdd/ReadVariableOp2\
,sequential_34/dense_34/MatMul/ReadVariableOp,sequential_34/dense_34/MatMul/ReadVariableOp2^
-sequential_34/dense_35/BiasAdd/ReadVariableOp-sequential_34/dense_35/BiasAdd/ReadVariableOp2\
,sequential_34/dense_35/MatMul/ReadVariableOp,sequential_34/dense_35/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_sequential_35_layer_call_and_return_conditional_losses_592452
sequential_34_input
sequential_34_592429
sequential_34_592431
sequential_34_592433
sequential_34_592435
identity��%sequential_34/StatefulPartitionedCall�
%sequential_34/StatefulPartitionedCallStatefulPartitionedCallsequential_34_inputsequential_34_592429sequential_34_592431sequential_34_592433sequential_34_592435*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_5923602'
%sequential_34/StatefulPartitionedCall�
softmax_17/PartitionedCallPartitionedCall.sequential_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_softmax_17_layer_call_and_return_conditional_losses_5924432
softmax_17/PartitionedCall�
IdentityIdentity#softmax_17/PartitionedCall:output:0&^sequential_34/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2N
%sequential_34/StatefulPartitionedCall%sequential_34/StatefulPartitionedCall:d `
/
_output_shapes
:���������
-
_user_specified_namesequential_34_input
�
�
.__inference_sequential_34_layer_call_fn_592661

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_5923602
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_sequential_34_layer_call_fn_592371
flatten_17_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallflatten_17_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_5923602
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:���������
*
_user_specified_nameflatten_17_input
�	
�
D__inference_dense_34_layer_call_and_return_conditional_losses_592706

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
F__inference_flatten_17_layer_call_and_return_conditional_losses_592690

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_sequential_35_layer_call_fn_592494
sequential_34_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallsequential_34_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_35_layer_call_and_return_conditional_losses_5924832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
/
_output_shapes
:���������
-
_user_specified_namesequential_34_input
�
�
.__inference_sequential_35_layer_call_fn_592589

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_35_layer_call_and_return_conditional_losses_5924832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_sequential_34_layer_call_and_return_conditional_losses_592360

inputs
dense_34_592349
dense_34_592351
dense_35_592354
dense_35_592356
identity�� dense_34/StatefulPartitionedCall� dense_35/StatefulPartitionedCall�
flatten_17/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_5922652
flatten_17/PartitionedCall�
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_34_592349dense_34_592351*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_5922842"
 dense_34/StatefulPartitionedCall�
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_592354dense_35_592356*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_5923102"
 dense_35/StatefulPartitionedCall�
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_sequential_34_layer_call_and_return_conditional_losses_592342
flatten_17_input
dense_34_592331
dense_34_592333
dense_35_592336
dense_35_592338
identity�� dense_34/StatefulPartitionedCall� dense_35/StatefulPartitionedCall�
flatten_17/PartitionedCallPartitionedCallflatten_17_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_5922652
flatten_17/PartitionedCall�
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_34_592331dense_34_592333*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_5922842"
 dense_34/StatefulPartitionedCall�
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_592336dense_35_592338*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_5923102"
 dense_35/StatefulPartitionedCall�
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:a ]
/
_output_shapes
:���������
*
_user_specified_nameflatten_17_input
�	
�
D__inference_dense_35_layer_call_and_return_conditional_losses_592725

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�[
�

"__inference__traced_restore_592893
file_prefix$
 assignvariableop_dense_34_kernel$
 assignvariableop_1_dense_34_bias&
"assignvariableop_2_dense_35_kernel$
 assignvariableop_3_dense_35_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count
assignvariableop_11_total_1
assignvariableop_12_count_1.
*assignvariableop_13_adam_dense_34_kernel_m,
(assignvariableop_14_adam_dense_34_bias_m.
*assignvariableop_15_adam_dense_35_kernel_m,
(assignvariableop_16_adam_dense_35_bias_m.
*assignvariableop_17_adam_dense_34_kernel_v,
(assignvariableop_18_adam_dense_34_bias_v.
*assignvariableop_19_adam_dense_35_kernel_v,
(assignvariableop_20_adam_dense_35_bias_v
identity_22��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBWvariables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_dense_34_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_34_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_35_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_35_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_dense_34_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_dense_34_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_35_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_35_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_34_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_34_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_35_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_35_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_209
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_21�
Identity_22IdentityIdentity_21:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_22"#
identity_22Identity_22:output:0*i
_input_shapesX
V: :::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
I__inference_sequential_35_layer_call_and_return_conditional_losses_592483

inputs
sequential_34_592472
sequential_34_592474
sequential_34_592476
sequential_34_592478
identity��%sequential_34/StatefulPartitionedCall�
%sequential_34/StatefulPartitionedCallStatefulPartitionedCallinputssequential_34_592472sequential_34_592474sequential_34_592476sequential_34_592478*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_5923602'
%sequential_34/StatefulPartitionedCall�
softmax_17/PartitionedCallPartitionedCall.sequential_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_softmax_17_layer_call_and_return_conditional_losses_5924432
softmax_17/PartitionedCall�
IdentityIdentity#softmax_17/PartitionedCall:output:0&^sequential_34/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2N
%sequential_34/StatefulPartitionedCall%sequential_34/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_softmax_17_layer_call_and_return_conditional_losses_592443

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
~
)__inference_dense_35_layer_call_fn_592734

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_5923102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
F__inference_flatten_17_layer_call_and_return_conditional_losses_592265

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_sequential_34_layer_call_fn_592674

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_5923882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_592536
sequential_34_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallsequential_34_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_5922552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
/
_output_shapes
:���������
-
_user_specified_namesequential_34_input
�4
�
__inference__traced_save_592820
file_prefix.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_34_kernel_m_read_readvariableop3
/savev2_adam_dense_34_bias_m_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableop5
1savev2_adam_dense_34_kernel_v_read_readvariableop3
/savev2_adam_dense_34_bias_v_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBWvariables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_34_kernel_m_read_readvariableop/savev2_adam_dense_34_bias_m_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableop1savev2_adam_dense_34_kernel_v_read_readvariableop/savev2_adam_dense_34_bias_v_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:	�
:
: : : : : : : : : :
��:�:	�
:
:
��:�:	�
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�
: 

_output_shapes
:
:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�
: 

_output_shapes
:
:

_output_shapes
: 
�	
�
D__inference_dense_34_layer_call_and_return_conditional_losses_592284

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_sequential_35_layer_call_and_return_conditional_losses_592510

inputs
sequential_34_592499
sequential_34_592501
sequential_34_592503
sequential_34_592505
identity��%sequential_34/StatefulPartitionedCall�
%sequential_34/StatefulPartitionedCallStatefulPartitionedCallinputssequential_34_592499sequential_34_592501sequential_34_592503sequential_34_592505*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_5923882'
%sequential_34/StatefulPartitionedCall�
softmax_17/PartitionedCallPartitionedCall.sequential_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_softmax_17_layer_call_and_return_conditional_losses_5924432
softmax_17/PartitionedCall�
IdentityIdentity#softmax_17/PartitionedCall:output:0&^sequential_34/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2N
%sequential_34/StatefulPartitionedCall%sequential_34/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
G
+__inference_softmax_17_layer_call_fn_592684

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_softmax_17_layer_call_and_return_conditional_losses_5924432
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
I__inference_sequential_34_layer_call_and_return_conditional_losses_592648

inputs+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity��dense_34/BiasAdd/ReadVariableOp�dense_34/MatMul/ReadVariableOp�dense_35/BiasAdd/ReadVariableOp�dense_35/MatMul/ReadVariableOpu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
flatten_17/Const�
flatten_17/ReshapeReshapeinputsflatten_17/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_17/Reshape�
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_34/MatMul/ReadVariableOp�
dense_34/MatMulMatMulflatten_17/Reshape:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_34/MatMul�
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_34/BiasAdd/ReadVariableOp�
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_34/BiasAddt
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_34/Relu�
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02 
dense_35/MatMul/ReadVariableOp�
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_35/MatMul�
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_35/BiasAdd/ReadVariableOp�
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_35/BiasAdd�
IdentityIdentitydense_35/BiasAdd:output:0 ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_sequential_35_layer_call_fn_592602

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_35_layer_call_and_return_conditional_losses_5925102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_sequential_35_layer_call_and_return_conditional_losses_592466
sequential_34_input
sequential_34_592455
sequential_34_592457
sequential_34_592459
sequential_34_592461
identity��%sequential_34/StatefulPartitionedCall�
%sequential_34/StatefulPartitionedCallStatefulPartitionedCallsequential_34_inputsequential_34_592455sequential_34_592457sequential_34_592459sequential_34_592461*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_5923882'
%sequential_34/StatefulPartitionedCall�
softmax_17/PartitionedCallPartitionedCall.sequential_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_softmax_17_layer_call_and_return_conditional_losses_5924432
softmax_17/PartitionedCall�
IdentityIdentity#softmax_17/PartitionedCall:output:0&^sequential_34/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2N
%sequential_34/StatefulPartitionedCall%sequential_34/StatefulPartitionedCall:d `
/
_output_shapes
:���������
-
_user_specified_namesequential_34_input
�
�
.__inference_sequential_34_layer_call_fn_592399
flatten_17_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallflatten_17_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_5923882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:���������
*
_user_specified_nameflatten_17_input
�
�
I__inference_sequential_34_layer_call_and_return_conditional_losses_592388

inputs
dense_34_592377
dense_34_592379
dense_35_592382
dense_35_592384
identity�� dense_34/StatefulPartitionedCall� dense_35/StatefulPartitionedCall�
flatten_17/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_5922652
flatten_17/PartitionedCall�
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_34_592377dense_34_592379*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_5922842"
 dense_34/StatefulPartitionedCall�
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_592382dense_35_592384*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_5923102"
 dense_35/StatefulPartitionedCall�
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
G
+__inference_flatten_17_layer_call_fn_592695

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_5922652
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_sequential_34_layer_call_and_return_conditional_losses_592327
flatten_17_input
dense_34_592295
dense_34_592297
dense_35_592321
dense_35_592323
identity�� dense_34/StatefulPartitionedCall� dense_35/StatefulPartitionedCall�
flatten_17/PartitionedCallPartitionedCallflatten_17_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_5922652
flatten_17/PartitionedCall�
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_34_592295dense_34_592297*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_5922842"
 dense_34/StatefulPartitionedCall�
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_592321dense_35_592323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_5923102"
 dense_35/StatefulPartitionedCall�
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:a ]
/
_output_shapes
:���������
*
_user_specified_nameflatten_17_input
�
b
F__inference_softmax_17_layer_call_and_return_conditional_losses_592679

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�$
�
!__inference__wrapped_model_592255
sequential_34_inputG
Csequential_35_sequential_34_dense_34_matmul_readvariableop_resourceH
Dsequential_35_sequential_34_dense_34_biasadd_readvariableop_resourceG
Csequential_35_sequential_34_dense_35_matmul_readvariableop_resourceH
Dsequential_35_sequential_34_dense_35_biasadd_readvariableop_resource
identity��;sequential_35/sequential_34/dense_34/BiasAdd/ReadVariableOp�:sequential_35/sequential_34/dense_34/MatMul/ReadVariableOp�;sequential_35/sequential_34/dense_35/BiasAdd/ReadVariableOp�:sequential_35/sequential_34/dense_35/MatMul/ReadVariableOp�
,sequential_35/sequential_34/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2.
,sequential_35/sequential_34/flatten_17/Const�
.sequential_35/sequential_34/flatten_17/ReshapeReshapesequential_34_input5sequential_35/sequential_34/flatten_17/Const:output:0*
T0*(
_output_shapes
:����������20
.sequential_35/sequential_34/flatten_17/Reshape�
:sequential_35/sequential_34/dense_34/MatMul/ReadVariableOpReadVariableOpCsequential_35_sequential_34_dense_34_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02<
:sequential_35/sequential_34/dense_34/MatMul/ReadVariableOp�
+sequential_35/sequential_34/dense_34/MatMulMatMul7sequential_35/sequential_34/flatten_17/Reshape:output:0Bsequential_35/sequential_34/dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+sequential_35/sequential_34/dense_34/MatMul�
;sequential_35/sequential_34/dense_34/BiasAdd/ReadVariableOpReadVariableOpDsequential_35_sequential_34_dense_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;sequential_35/sequential_34/dense_34/BiasAdd/ReadVariableOp�
,sequential_35/sequential_34/dense_34/BiasAddBiasAdd5sequential_35/sequential_34/dense_34/MatMul:product:0Csequential_35/sequential_34/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2.
,sequential_35/sequential_34/dense_34/BiasAdd�
)sequential_35/sequential_34/dense_34/ReluRelu5sequential_35/sequential_34/dense_34/BiasAdd:output:0*
T0*(
_output_shapes
:����������2+
)sequential_35/sequential_34/dense_34/Relu�
:sequential_35/sequential_34/dense_35/MatMul/ReadVariableOpReadVariableOpCsequential_35_sequential_34_dense_35_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02<
:sequential_35/sequential_34/dense_35/MatMul/ReadVariableOp�
+sequential_35/sequential_34/dense_35/MatMulMatMul7sequential_35/sequential_34/dense_34/Relu:activations:0Bsequential_35/sequential_34/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2-
+sequential_35/sequential_34/dense_35/MatMul�
;sequential_35/sequential_34/dense_35/BiasAdd/ReadVariableOpReadVariableOpDsequential_35_sequential_34_dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02=
;sequential_35/sequential_34/dense_35/BiasAdd/ReadVariableOp�
,sequential_35/sequential_34/dense_35/BiasAddBiasAdd5sequential_35/sequential_34/dense_35/MatMul:product:0Csequential_35/sequential_34/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2.
,sequential_35/sequential_34/dense_35/BiasAdd�
 sequential_35/softmax_17/SoftmaxSoftmax5sequential_35/sequential_34/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2"
 sequential_35/softmax_17/Softmax�
IdentityIdentity*sequential_35/softmax_17/Softmax:softmax:0<^sequential_35/sequential_34/dense_34/BiasAdd/ReadVariableOp;^sequential_35/sequential_34/dense_34/MatMul/ReadVariableOp<^sequential_35/sequential_34/dense_35/BiasAdd/ReadVariableOp;^sequential_35/sequential_34/dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2z
;sequential_35/sequential_34/dense_34/BiasAdd/ReadVariableOp;sequential_35/sequential_34/dense_34/BiasAdd/ReadVariableOp2x
:sequential_35/sequential_34/dense_34/MatMul/ReadVariableOp:sequential_35/sequential_34/dense_34/MatMul/ReadVariableOp2z
;sequential_35/sequential_34/dense_35/BiasAdd/ReadVariableOp;sequential_35/sequential_34/dense_35/BiasAdd/ReadVariableOp2x
:sequential_35/sequential_34/dense_35/MatMul/ReadVariableOp:sequential_35/sequential_34/dense_35/MatMul/ReadVariableOp:d `
/
_output_shapes
:���������
-
_user_specified_namesequential_34_input
�
�
I__inference_sequential_35_layer_call_and_return_conditional_losses_592556

inputs9
5sequential_34_dense_34_matmul_readvariableop_resource:
6sequential_34_dense_34_biasadd_readvariableop_resource9
5sequential_34_dense_35_matmul_readvariableop_resource:
6sequential_34_dense_35_biasadd_readvariableop_resource
identity��-sequential_34/dense_34/BiasAdd/ReadVariableOp�,sequential_34/dense_34/MatMul/ReadVariableOp�-sequential_34/dense_35/BiasAdd/ReadVariableOp�,sequential_34/dense_35/MatMul/ReadVariableOp�
sequential_34/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2 
sequential_34/flatten_17/Const�
 sequential_34/flatten_17/ReshapeReshapeinputs'sequential_34/flatten_17/Const:output:0*
T0*(
_output_shapes
:����������2"
 sequential_34/flatten_17/Reshape�
,sequential_34/dense_34/MatMul/ReadVariableOpReadVariableOp5sequential_34_dense_34_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,sequential_34/dense_34/MatMul/ReadVariableOp�
sequential_34/dense_34/MatMulMatMul)sequential_34/flatten_17/Reshape:output:04sequential_34/dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_34/dense_34/MatMul�
-sequential_34/dense_34/BiasAdd/ReadVariableOpReadVariableOp6sequential_34_dense_34_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_34/dense_34/BiasAdd/ReadVariableOp�
sequential_34/dense_34/BiasAddBiasAdd'sequential_34/dense_34/MatMul:product:05sequential_34/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
sequential_34/dense_34/BiasAdd�
sequential_34/dense_34/ReluRelu'sequential_34/dense_34/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_34/dense_34/Relu�
,sequential_34/dense_35/MatMul/ReadVariableOpReadVariableOp5sequential_34_dense_35_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02.
,sequential_34/dense_35/MatMul/ReadVariableOp�
sequential_34/dense_35/MatMulMatMul)sequential_34/dense_34/Relu:activations:04sequential_34/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
sequential_34/dense_35/MatMul�
-sequential_34/dense_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_34_dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_34/dense_35/BiasAdd/ReadVariableOp�
sequential_34/dense_35/BiasAddBiasAdd'sequential_34/dense_35/MatMul:product:05sequential_34/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2 
sequential_34/dense_35/BiasAdd�
softmax_17/SoftmaxSoftmax'sequential_34/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
softmax_17/Softmax�
IdentityIdentitysoftmax_17/Softmax:softmax:0.^sequential_34/dense_34/BiasAdd/ReadVariableOp-^sequential_34/dense_34/MatMul/ReadVariableOp.^sequential_34/dense_35/BiasAdd/ReadVariableOp-^sequential_34/dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2^
-sequential_34/dense_34/BiasAdd/ReadVariableOp-sequential_34/dense_34/BiasAdd/ReadVariableOp2\
,sequential_34/dense_34/MatMul/ReadVariableOp,sequential_34/dense_34/MatMul/ReadVariableOp2^
-sequential_34/dense_35/BiasAdd/ReadVariableOp-sequential_34/dense_35/BiasAdd/ReadVariableOp2\
,sequential_34/dense_35/MatMul/ReadVariableOp,sequential_34/dense_35/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_sequential_35_layer_call_fn_592521
sequential_34_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallsequential_34_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_35_layer_call_and_return_conditional_losses_5925102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
/
_output_shapes
:���������
-
_user_specified_namesequential_34_input"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
[
sequential_34_inputD
%serving_default_sequential_34_input:0���������>

softmax_170
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�!
layer_with_weights-0
layer-0
layer-1
	variables
regularization_losses
trainable_variables
	keras_api

signatures
*Z&call_and_return_all_conditional_losses
[_default_save_signature
\__call__"�
_tf_keras_sequential�{"class_name": "Sequential", "name": "sequential_35", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_35", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_34_input"}}, {"class_name": "Sequential", "config": {"name": "sequential_34", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_17_input"}}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 1000, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, {"class_name": "Softmax", "config": {"name": "softmax_17", "trainable": true, "dtype": "float32", "axis": -1}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_35", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_34_input"}}, {"class_name": "Sequential", "config": {"name": "sequential_34", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_17_input"}}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 1000, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, {"class_name": "Softmax", "config": {"name": "softmax_17", "trainable": true, "dtype": "float32", "axis": -1}}]}}}
�
layer-0
	layer_with_weights-0
	layer-1

layer_with_weights-1

layer-2
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api
*]&call_and_return_all_conditional_losses
^__call__"�
_tf_keras_sequential�{"class_name": "Sequential", "name": "sequential_34", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_34", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_17_input"}}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 1000, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_34", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_17_input"}}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 1000, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": true, "label_smoothing": 0}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�
	variables
regularization_losses
trainable_variables
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"�
_tf_keras_layer�{"class_name": "Softmax", "name": "softmax_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax_17", "trainable": true, "dtype": "float32", "axis": -1}}
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
�
	variables
non_trainable_variables
regularization_losses

layers
layer_metrics
trainable_variables
metrics
layer_regularization_losses
\__call__
[_default_save_signature
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
,
aserving_default"
signature_map
�
	variables
regularization_losses
trainable_variables
 	keras_api
*b&call_and_return_all_conditional_losses
c__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

kernel
bias
!	variables
"regularization_losses
#trainable_variables
$	keras_api
*d&call_and_return_all_conditional_losses
e__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 1000, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
�

kernel
bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
*f&call_and_return_all_conditional_losses
g__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}}
�
)iter

*beta_1

+beta_2
	,decay
-learning_ratemRmSmTmUvVvWvXvY"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
�
	variables
.non_trainable_variables
regularization_losses

/layers
0layer_metrics
trainable_variables
1metrics
2layer_regularization_losses
^__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
3non_trainable_variables
regularization_losses

4layers
5layer_metrics
trainable_variables
6metrics
7layer_regularization_losses
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
#:!
��2dense_34/kernel
:�2dense_34/bias
": 	�
2dense_35/kernel
:
2dense_35/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
8non_trainable_variables
regularization_losses

9layers
:layer_metrics
trainable_variables
;metrics
<layer_regularization_losses
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
!	variables
=non_trainable_variables
"regularization_losses

>layers
?layer_metrics
#trainable_variables
@metrics
Alayer_regularization_losses
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
%	variables
Bnon_trainable_variables
&regularization_losses

Clayers
Dlayer_metrics
'trainable_variables
Emetrics
Flayer_regularization_losses
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
5
0
	1

2"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	Itotal
	Jcount
K	variables
L	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�
	Mtotal
	Ncount
O
_fn_kwargs
P	variables
Q	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
I0
J1"
trackable_list_wrapper
-
K	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
M0
N1"
trackable_list_wrapper
-
P	variables"
_generic_user_object
(:&
��2Adam/dense_34/kernel/m
!:�2Adam/dense_34/bias/m
':%	�
2Adam/dense_35/kernel/m
 :
2Adam/dense_35/bias/m
(:&
��2Adam/dense_34/kernel/v
!:�2Adam/dense_34/bias/v
':%	�
2Adam/dense_35/kernel/v
 :
2Adam/dense_35/bias/v
�2�
I__inference_sequential_35_layer_call_and_return_conditional_losses_592576
I__inference_sequential_35_layer_call_and_return_conditional_losses_592556
I__inference_sequential_35_layer_call_and_return_conditional_losses_592452
I__inference_sequential_35_layer_call_and_return_conditional_losses_592466�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_592255�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *:�7
5�2
sequential_34_input���������
�2�
.__inference_sequential_35_layer_call_fn_592521
.__inference_sequential_35_layer_call_fn_592589
.__inference_sequential_35_layer_call_fn_592602
.__inference_sequential_35_layer_call_fn_592494�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
I__inference_sequential_34_layer_call_and_return_conditional_losses_592648
I__inference_sequential_34_layer_call_and_return_conditional_losses_592327
I__inference_sequential_34_layer_call_and_return_conditional_losses_592342
I__inference_sequential_34_layer_call_and_return_conditional_losses_592629�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
.__inference_sequential_34_layer_call_fn_592674
.__inference_sequential_34_layer_call_fn_592371
.__inference_sequential_34_layer_call_fn_592661
.__inference_sequential_34_layer_call_fn_592399�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_softmax_17_layer_call_and_return_conditional_losses_592679�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_softmax_17_layer_call_fn_592684�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_592536sequential_34_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_flatten_17_layer_call_and_return_conditional_losses_592690�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_flatten_17_layer_call_fn_592695�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_34_layer_call_and_return_conditional_losses_592706�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_34_layer_call_fn_592715�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_35_layer_call_and_return_conditional_losses_592725�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_35_layer_call_fn_592734�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_592255�D�A
:�7
5�2
sequential_34_input���������
� "7�4
2

softmax_17$�!

softmax_17���������
�
D__inference_dense_34_layer_call_and_return_conditional_losses_592706^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_dense_34_layer_call_fn_592715Q0�-
&�#
!�
inputs����������
� "������������
D__inference_dense_35_layer_call_and_return_conditional_losses_592725]0�-
&�#
!�
inputs����������
� "%�"
�
0���������

� }
)__inference_dense_35_layer_call_fn_592734P0�-
&�#
!�
inputs����������
� "����������
�
F__inference_flatten_17_layer_call_and_return_conditional_losses_592690a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������
� �
+__inference_flatten_17_layer_call_fn_592695T7�4
-�*
(�%
inputs���������
� "������������
I__inference_sequential_34_layer_call_and_return_conditional_losses_592327xI�F
?�<
2�/
flatten_17_input���������
p

 
� "%�"
�
0���������

� �
I__inference_sequential_34_layer_call_and_return_conditional_losses_592342xI�F
?�<
2�/
flatten_17_input���������
p 

 
� "%�"
�
0���������

� �
I__inference_sequential_34_layer_call_and_return_conditional_losses_592629n?�<
5�2
(�%
inputs���������
p

 
� "%�"
�
0���������

� �
I__inference_sequential_34_layer_call_and_return_conditional_losses_592648n?�<
5�2
(�%
inputs���������
p 

 
� "%�"
�
0���������

� �
.__inference_sequential_34_layer_call_fn_592371kI�F
?�<
2�/
flatten_17_input���������
p

 
� "����������
�
.__inference_sequential_34_layer_call_fn_592399kI�F
?�<
2�/
flatten_17_input���������
p 

 
� "����������
�
.__inference_sequential_34_layer_call_fn_592661a?�<
5�2
(�%
inputs���������
p

 
� "����������
�
.__inference_sequential_34_layer_call_fn_592674a?�<
5�2
(�%
inputs���������
p 

 
� "����������
�
I__inference_sequential_35_layer_call_and_return_conditional_losses_592452{L�I
B�?
5�2
sequential_34_input���������
p

 
� "%�"
�
0���������

� �
I__inference_sequential_35_layer_call_and_return_conditional_losses_592466{L�I
B�?
5�2
sequential_34_input���������
p 

 
� "%�"
�
0���������

� �
I__inference_sequential_35_layer_call_and_return_conditional_losses_592556n?�<
5�2
(�%
inputs���������
p

 
� "%�"
�
0���������

� �
I__inference_sequential_35_layer_call_and_return_conditional_losses_592576n?�<
5�2
(�%
inputs���������
p 

 
� "%�"
�
0���������

� �
.__inference_sequential_35_layer_call_fn_592494nL�I
B�?
5�2
sequential_34_input���������
p

 
� "����������
�
.__inference_sequential_35_layer_call_fn_592521nL�I
B�?
5�2
sequential_34_input���������
p 

 
� "����������
�
.__inference_sequential_35_layer_call_fn_592589a?�<
5�2
(�%
inputs���������
p

 
� "����������
�
.__inference_sequential_35_layer_call_fn_592602a?�<
5�2
(�%
inputs���������
p 

 
� "����������
�
$__inference_signature_wrapper_592536�[�X
� 
Q�N
L
sequential_34_input5�2
sequential_34_input���������"7�4
2

softmax_17$�!

softmax_17���������
�
F__inference_softmax_17_layer_call_and_return_conditional_losses_592679\3�0
)�&
 �
inputs���������


 
� "%�"
�
0���������

� ~
+__inference_softmax_17_layer_call_fn_592684O3�0
)�&
 �
inputs���������


 
� "����������
