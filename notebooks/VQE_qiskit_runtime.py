from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options
from qiskit.algorithms.minimum_eigensolvers import VQE
import numpy as np
from qiskit.algorithms.optimizers import SLSQP, L_BFGS_B, SPSA, COBYLA
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter, ParityMapper
from qiskit_nature.second_q.properties import ParticleNumber
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_ibm_provider import IBMProvider

provider = IBMProvider()
bond_distance = 0.28
driver = PySCFDriver(
    atom=f"H 0 0 0; H 0 0 {bond_distance}",
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)
es_problem = driver.run()
hamiltonian = es_problem.hamiltonian
second_q_op = hamiltonian.second_q_op()
converter = QubitConverter(
    ParityMapper(), two_qubit_reduction=True, z2symmetry_reduction=[-1])
num_particles = es_problem.num_particles
num_spatial_orbitals = es_problem.num_spatial_orbitals
qubit_op = converter.convert(second_q_op, num_particles)
init_state = HartreeFock(num_spatial_orbitals, num_particles, converter)
ansatz = UCCSD(num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles,
               qubit_converter=converter, reps=1, initial_state=init_state, generalized=False)
service = QiskitRuntimeService(
    channel="ibm_quantum", instance='ibm-q-education/ibm-india-1/acm-winter-schoo')
sim = service.get_backend('ibmq_qasm_simulator')
backend = service.get_backend('ibmq_jakarta')
noise_model = NoiseModel.from_backend(backend)
options = Options()
options.simulator = {
    "noise_model": noise_model,
    "basis_gates": backend.configuration().basis_gates,
    "coupling_map": backend.configuration().coupling_map
}
options.execution.shots = 500000
options.optimization_level = 0
options.resilience_level = 1
optimizer = L_BFGS_B(maxiter=100, eps=1e-02)  # for a quick test
with Session(service=service, backend=sim):
    estimator = Estimator(options=options)
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
    result = vqe.compute_minimum_eigenvalue(qubit_op)
print(result)
freeze_core = es_problem.hamiltonian.constants
energies = list(freeze_core.values())
gse = result.optimal_value + energies[0]
print("gse(hartree)=", gse)
