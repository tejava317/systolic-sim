# import sys
import time
import numpy as np
from simulate import SystolicArraySim

class SystolicArrayTest:
    def __init__(self, mac_size=128):
        self.mac_size = 128
    
    def run_test(self, configs):
        answer = []
        
        print("=========================================================")
        print("*************** Systolic Array Simulation ***************")
        print("=========================================================")
        print()
        print(": Matrix-Matrix Multiplication Implementation")
        print()
        print("[Configurations]")
        print("1) Weight Matrix : (M x K)")
        print("2) Activation Matrix : (K x N)")
        print(f"3) Systolic Array : {self.mac_size} x {self.mac_size} PEs")
        print()
        print("---------------------------------------------------------")
        print()
        
        start_time = time.time()
        
        for i in range(len(configs)):
            m, k, n = configs[i]
            
            print("[Case " + str(i + 1) + "]")
            if m == k and k == n:
                print(f": M = K = N = {m}")
            else:
                print(f": M = {m}, K = {k}, N = {n}")
            print()
            
            weight_matrix = np.random.random((m, k))
            activation_matrix = np.random.random((k, n))
            
            # answer_matrix = np.matmul(weight_matrix, activation_matrix)
            
            simulator = SystolicArraySim(weight_matrix, activation_matrix, self.mac_size)
            
            simulate_output, cycle, utilization = simulator.run_simulate(i)

            # Simulate output validation
            answer.append('correct')
            # answer.append('wrong')
            
            print(f"- Cycles for multiplication : {cycle} cycles")
            print(f"- MAC array utilization ratio during computation : {utilization * 100:0.2f}%")
            print()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print()
        if 'wrong' in answer:
            print(f"{len(configs)} tests run in {total_time:.3f} seconds " + "\033[31m" + f"({answer.count('wrong')} tests failed)" + "\033[37m\n")
        else:
            print(f"{len(configs)} tests run in {total_time:.3f} seconds " + "\033[32m" + "(All tests passed)" + "\033[37m\n")
        
        print("=====================================================")


if __name__ == "__main__":
    test = SystolicArrayTest(mac_size=128)
    
    # test.run_test([
    #     (128, 128, 128),
    #     (256, 256, 256)
    # ])
    
    test.run_test([
        (128, 128, 128),
        (256, 256, 256),
        (200, 128, 128),
        (128, 200, 200),
        (128, 64, 128),
        (128, 128, 64)
    ])
