from qiskit.circuit.library import ZZFeatureMap, TwoLocal, PauliFeatureMap, RealAmplitudes, EfficientSU2
import matplotlib.pyplot as plt

n_features = 2

# feature map and ansatz definition
feature_map_zz_1 = ZZFeatureMap(n_features, reps=1)
feature_map_zz_3 = ZZFeatureMap(n_features, reps=3)
feature_map_pauli = PauliFeatureMap(n_features, reps=1, paulis=['ZZ', 'ZX', 'ZY', 'XY'])

ansatz_tl_3 = TwoLocal(n_features, ['ry', 'rz'], 'cz', reps=3)
ansatz_ra_3 = RealAmplitudes(num_qubits=n_features, reps=3)
ansatz_esu2 = EfficientSU2(n_features, su2_gates=['rx', 'y'], entanglement='circular', reps=1)

optimizers = ['SPSA', 'QN-SPSA', 'COBYLA', 'NM']

feature_maps = [feature_map_zz_1, feature_map_zz_3, feature_map_pauli]
ansatze = [ansatz_ra_3, ansatz_tl_3, ansatz_esu2]

for fm in feature_maps:
    fm.decompose().draw(output='mpl')
    name = fm.name
    plt.savefig('img/circuits/{}'.format(name))


for a in ansatze:
    a.decompose().draw(output='mpl')
    name = a.name
    plt.savefig('img/circuits/{}'.format(name))

i = 1
for fm in feature_maps:
    for a in ansatze:
        for opt in optimizers:
            print('CONFIGURATION NUMBER {}'.format(i))
            fm = fm.decompose()
            print('FEATURE MAP')
            print(fm.draw())
            print('name: {}'.format(fm.name))
            print('#parameters: {}'.format(fm.num_parameters))
            print('gates: {}'.format(dict(fm.count_ops())))
            print('#gates: {}'.format(sum(list(dict(fm.count_ops()).values()))))
            print('depth: {}'.format(fm.depth()))
            a = a.decompose()
            print('ANSATZ')
            print(a.draw())
            print('name: {}'.format(a.name))
            print('#parameters: {}'.format(a.num_parameters))
            print('gates: {}'.format(dict(a.count_ops())))
            print('#gates: {}'.format(sum(list(dict(a.count_ops()).values()))))
            print('depth: {}'.format(a.depth()))
            # merges feature map and ansatz to make the complete circuit
            circuit = fm.compose(a)
            print(circuit.draw())
            print('#parameters: {}'.format(circuit.num_parameters))
            print('gates: {}'.format(dict(circuit.count_ops())))
            print('#gates: {}'.format(sum(list(dict(circuit.count_ops()).values()))))
            print('depth: {}'.format(circuit.depth()))
            print('OPTIMIZER: {}'.format(opt))
            print('-----------------------------------------------------------------------')
            i += 1
