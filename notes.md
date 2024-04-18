# Rejected

## use(QC) for matmult
Matmult cannot be done because for some arbitrary matrix $A$ we can do
$UDV^* = A$ such that $U,V$ are unitary and the maybe even break $D$ down to unitaries so finally we get

$A = U (\sum_i D_i) V^* = \sum_i U D_i V^*$ which means we can apply each $UD_iV^*$ as a seperate gate as $e^{iUD_iV^*}$. But the problem is we cannot do each $U,D_i,V^*$ as a seperate gate which is what is required since quantum gates are unitary.

The way to get around it may be to use trotterization but we go back to square one of the existing methods.

# General
## On QML
- Using QML to represent real data comes from the belief that Hilbert space has a better "search space". This belief is fundamentally flawed since it's merely Cauchy complete and adding $e^x$ to our list of available functions doesn't really add much to the search space since we already have polynomials which can do approximated $e^x$
- Once in the Hilbert space everything is linear (in QC). So unless we have external non-linearities injected, like re-encoding and using that as $R_x, R_y$ there is no reason to use QML over classical ML since any level of complexity can be reduced down to one $H-R_x-R_y$ op per qubit. [Somewhere in here](https://arxiv.org/pdf/2101.11020)