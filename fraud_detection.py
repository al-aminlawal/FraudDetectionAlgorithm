from VEA import *
from Factor import Factor
import copy
factor_Fraud = Factor(['Fraud'],['Trav'],[0.01,0.004,0.99,0.996])

factor_Trav = Factor(['Trav'],[],[0.05,0.95])

factor_FP = Factor(['FP'],['Trav','Fraud'],[0.9,0.1,0.9,0.1,0.1,0.9,0.01,0.99])

factor_CRP = Factor(['CRP'],['OC'],[0.1,0.9,0.001,0.009])

factor_OC = Factor(['OC'],[],[0.7,0.3])

factor_IP = Factor(['IP'],['Fraud','CRP'],[0.02,0.01,0.011,0.001,0.98,0.99,0.989,0.999])


# answer to questions 1, when nothing is observed
FacList1 = [copy.deepcopy(factor_Fraud), copy.deepcopy(factor_Trav), copy.deepcopy(factor_FP), copy.deepcopy(factor_CRP), copy.deepcopy(factor_OC),copy.deepcopy(factor_IP)]
hidList1 = ['Trav','FP','OC','CRP','IP']
evidDict1 = {}
queryVars1 = ['Fraud']

resultFactor(FacList1, queryVars1, hidList1, evidDict1)

# answer to questions 2, when +FP, -IP +CRP
FacList2 = [copy.deepcopy(factor_Fraud), copy.deepcopy(factor_Trav), copy.deepcopy(factor_FP), copy.deepcopy(factor_CRP), copy.deepcopy(factor_OC),copy.deepcopy(factor_IP)]
hidList2 = ['Trav','OC']
evidDict2 = {'FP': True, 'IP': False,'CRP':True}
queryVars2 = ['Fraud']

resultFactor(FacList2, queryVars2, hidList2, evidDict2)


