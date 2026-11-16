======================
Tutorial: Broadcasting
======================

Arraymancer supports explicit broadcasting with ``broadcast`` and its
alias ``bc``. And supports implicit broadcasting with operations
beginning with a dot:

.. code:: nim

    let j = [0, 10, 20, 30].toTensor.reshape(4,1)
    let k = [0, 1, 2].toTensor.reshape(1,3)

    echo j +. k
    # Tensor[int] of shape "[4, 3]" on backend "Cpu"
    # |0       1       2|
    # |10     11      12|
    # |20     21      22|
    # |30     31      32|

Tensor-tensor operations align dimensions from the right. A missing leading
dimension acts like a dimension of size 1, so a tensor of shape ``[2, 3]`` can
be added to one of shape ``[3]``. Non-singleton trailing dimensions must match.

-  ``+.``,\ ``-.``,
-  ``*.``: broadcasted element-wise matrix multiplication also called
   Hadamard product)
-  ``./``: broadcasted element-wise division or integer-division
-  ``+.=``, ``-.=``, ``*.=``, ``/.=``: in-place versions. Only the right
   operand is broadcastable.
