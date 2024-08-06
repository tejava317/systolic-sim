# import sys
import time
import numpy as np
from simulate import SystolicArraySim

class SystolicArrayTest:
    def __init__(self, mac_size=128):
        self.mac_size = 128
    
    def run_test(self, configs):
        print("=====================================================")
        print("************* Systolic Array Simulation *************")
        print("=====================================================")
        print()
        
        start_time = time.time()
        
        for i in range(len(configs)):
            m, k, n = configs[i]
            
            print("[Case " + str(i + 1) + "]")
            if m == k and k == n:
                print(f": M = K = N = {m}")
            else:
                print(f": M = {m}, K = {k}, N = {n}")
            
            weight_matrix = np.random.random((m, k))
            activation_matrix = np.random.random((k, n))
            
            # answer_matrix = np.matmul(weight_matrix, activation_matrix)
            
            simulator = SystolicArraySim(weight_matrix, activation_matrix, self.mac_size)
            
            cycle = simulator.run_simulate(i)

            print(f"Cycles for computation : {cycle} cycles\n")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"{len(configs)} tests run in {total_time:.3f} seconds " + "\033[32m" + f"({len(configs)} tests passed)" + "\033[37m\n")
        
        print("=====================================================")


if __name__ == "__main__":
    test = SystolicArrayTest(mac_size=128)
    
    # test.run_test([
    #     (128, 128, 128)
    # ])
    
    test.run_test([
        (128, 128, 128),
        (256, 256, 256),
        (200, 128, 128),
        (128, 200, 200),
    ])
