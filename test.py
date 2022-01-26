from VEA import *
from Factor import Factor

#####################inference test code with Unit 4 page 18 example
# factor_b = Factor(['b'], [], [0.001, 0.999])
# factor_e = Factor(['e'], [], [0.002, 0.998])
# factor_j = Factor(['j'], ['a'], [0.9, 0.1, 0.05, 0.95])
# factor_m = Factor(['m'], ['a'], [0.7, 0.3, 0.01, 0.99])
# factor_a = Factor(['a'], ['b', 'e'], [0.95, 0.05, 0.94, 0.06, 0.29, 0.71, 0.001, 0.999])
#
# FacList = [factor_a, factor_b, factor_e, factor_j, factor_m]
# hidList = ['a', 'e']
# evidDict = {'j': True, 'm': True}
# queryVars = ['b']
#
# resultFactor(FacList, queryVars, hidList, evidDict)

##################### other test code for functions

# factor1 = Factor(['w'],[],[0.8,0.2])
# factor2 = Factor(['d'],['w'],[0.1,0.9,0.7,0.3])
# factor3 = productFactor(factor1,factor2)
# factor3.print_factor()
# factor1.print_factor()
# factor2.print_factor()
# factor_j = Factor(['j'], ['a'], [0.9, 0.1, 0.05, 0.95])
# factor_1 = Factor(['a'], ['b', 'e'], [0.95, 0.05, 0.94, 0.06, 0.29, 0.71, 0.001, 0.999])
# factor_2 = Factor(['a'], ['b', 'e'], [0.95, 0.05, 0.94, 0.06, 0.29, 0.71, 0.001, 0.999])
# # factor_1.print_factor()
# VEA.sumFactor(factor_1, 'b')
# # factor_1.print_factor()
# VEA.sumFactor(factor_2, 'e')
# factor_2.print_factor()
# factor_3 = VEA.productFactor(factor_1, factor_2)
# factor_3.print_factor()
# factor_1.print_factor()
# factor_2.print_factor()
# factor_4 = Factor(['a'], ['b', 'e'], [0.95, 0.05, 0.94, 0.06, 0.29, 0.71, 0.001, 0.999])
# factor_4.print_factor()

# VEA.observeFactor(factor_j,'j',True)
# factor_j.print_factor()
# factor_a.print_factor()
# newFactor = VEA.productFactor(factor_a,factor_j)
# newFactor.print_factor()
# factor2 = Factor(['m'], ['a'], [0.7, 0.3, 0.01, 0.99])
# factor2.print_factor()
# VEA.sumFactor(factor2,'m')
# factor2.print_factor()
# factor1 = Factor(['a'], ['b', 'e'], [0.95, 0.05, 0.94, 0.06, 0.29, 0.71, 0.001, 0.999])
# factor2 = Factor(['b'],[],[0.001,0.999])
# newFactor = productFactor(factor1, factor2)
# factor1.print_factor()
# factor2.print_factor()
# newFactor.print_factor()
# factor3 = Factor(['e'],[],[0.002,0.998])
# newFactor2 = productFactor(newFactor,factor3)
# newFactor2.print_factor()
# factor1 = Factor(['e'], [], [0.002, 0.998])
# VEA.sumFactor(factor1, 'e')
# factor1.print_factor()
# factor1 = Factor(['m'], ['a'], [0.7, 0.3, 0.01, 0.99])
# factor2 = Factor(['a'], ['b', 'e'], [0.95, 0.05, 0.94, 0.06, 0.29, 0.71, 0.001, 0.999])
# factor4 = Factor(['e'], [], [0.002, 0.998])
# factor3 = Factor(['a'], ['b', 'e'], [0.95, 0.05, 0.94, 0.06, 0.29, 0.71, 0.001, 0.999])
# productFactor(factor4, factor3)
# test code for Factor class
# factor1 = Factor(['a', 'b'], [0.3, 0.7, 0.4, 0.6])
# factor1.print_factor()
# factor2 = Factor(['c'], [0.5, 0.5])
# factor2.print_factor()
# factor3 = Factor(['a'], ['b', 'e'], [0.95, 0.05, 0.94, 0.06, 0.29, 0.71, 0.001, 0.999])
# factor3.print_factor()
# print(factor3.is_conditional())
# observeFactor(factor3, 'b', True)
# factor3.print_factor()
# sumFactor(factor3, 'b')
# factor3.print_factor()
# normalizedFactor(factor3)
# factor3.print_factor()
# sumFactor(factor3, 'e')
# print(factor3.is_conditional())
# factor3.print_factor()

# observeFactor(factor3, 'e', True)
# factor3.print_factor()
# normalizedFactor(factor3)
# factor3.print_factor()
# factor4 = Factor(['t', 'w'], [0.4, 0.1, 0.2, 0.3])
# test code for observeFactor
# print("\n test observe 'e' as True \n")
# observeFactor(factor3, 'e', True)
# factor3.print_factor()
#
# observeFactor(factor2, 'c', False)
# factor2.print_factor()

# test code for result factor
# sumFactor(factor3, 'e')
# factor3.print_factor()
# sumFactor(factor1,'b')
# sumFactor(factor4,'t')
# factor3.print_factor()
# normalizedFactor(factor3)
# factor3.print_factor()
