## Rejected

Matmult cannot be done because for some arbitrary matrix $A$ we can do
$UDV^* = A$ such that $U,V$ are unitary and the maybe even break $D$ down to unitaries so finally we get

$A = U (\sum_i D_i) V^* = \sum_i U D_i V^*$ which means we can apply each $UD_iV^*$ as a seperate gate as $e^{iUD_iV^*}$. But the problem is we cannot do each $U,D_i,V^*$ as a seperate gate which is what is required since quantum gates are unitary.

The way to get around it may be to use trotterization but we go back to square one of the existing methods.