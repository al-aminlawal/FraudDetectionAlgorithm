import numpy
from Factor import Factor


def observeFactor(factor, variable, value):
    if variable not in factor.variable_list:
        print("observe ALERT: variable not in factor")
        return
    variable_col_in_table = factor.variable_list.index(variable)
    remove_index = numpy.where(factor.table[:, variable_col_in_table] != value)[0].tolist()
    new_table = factor.table[factor.table[:, variable_col_in_table] == value]
    new_prob_list = [i for j, i in enumerate(factor.prob_list) if j not in remove_index]
    factor.table = new_table
    factor.prob_list = new_prob_list
    factor.observed_var_dict.update({variable: value})


def sumFactor(factor, variable):
    if variable not in factor.variable_list:
        print("sum ALERT: variable not in factor")
        return
    index = factor.variable_list.index(variable)
    factor.sum_var(variable)
    new_table = numpy.delete(factor.table, index, 1)
    unique_table, unique_index = numpy.unique(new_table, axis=0, return_index=True)
    new_prob_list = []
    for i, value in enumerate(unique_index):
        prob = factor.prob_list[value]
        for j in range(len(factor.prob_list)):
            if j != value:
                if (new_table[value] == new_table[j]).all():
                    prob += factor.prob_list[j]
        new_prob_list.append(prob)
    new_prob_list.reverse()
    unique_table = numpy.flip(unique_table, 0)
    factor.prob_list = new_prob_list
    factor.table = unique_table


def normalizedFactor(factor):
    if factor.father_list:
        print("ALERT: normalize is useful only when the factor is a distribution")
        return
    sum_prob = sum(factor.prob_list)
    for i, value in enumerate(factor.prob_list):
        factor.prob_list[i] = value / sum_prob


# P(x|y) * P(y) = P(x,y)
# input: two Factor
# return: new Factor
def productFactor(factor1, factor2):
    common_var = list(set(factor1.variable_list) & set(factor2.variable_list))
    # identify
    if set(common_var).issubset(set(factor1.son_list)) and set(common_var).issubset(set(factor2.father_list)):
        factor1, factor2 = factor2, factor1
    new_son_list = factor1.son_list + factor2.son_list
    new_son_list = list(dict.fromkeys(new_son_list))
    new_father_list = factor2.father_list + [x for x in factor1.father_list if x not in common_var]
    new_variable_list = new_father_list + new_son_list
    new_prob_list = []
    cols_in_2 = [i for i, x in enumerate(factor2.variable_list) if x in common_var]
    cols_in_1 = [i for i, x in enumerate(factor1.variable_list) if x in common_var]
    empty_prob_list = [0] * pow(2, len(new_variable_list))
    for i, i_value in enumerate(factor2.table):
        for j, j_value in enumerate(factor1.table):
            if (i_value[cols_in_2] == j_value[cols_in_1]).all():
                string = ''
                for variable in new_father_list + new_son_list:
                    if variable in factor2.variable_list:
                        var_index = factor2.variable_list.index(variable)
                        string += str(int(i_value[var_index]))
                    else:
                        var_index = factor1.variable_list.index(variable)
                        string += str(int(j_value[var_index]))
                prob_index = len(empty_prob_list) - 1 - int(string, 2)
                empty_prob_list[prob_index] = factor2.prob_list[i] * factor1.prob_list[j]
    if not factor1.observed_var_dict and not factor2.observed_var_dict:
        return Factor(new_son_list, new_father_list, empty_prob_list)
    else:
        observed_dict = factor1.observed_var_dict.copy()
        observed_dict.update(factor2.observed_var_dict)
        retFactor = Factor(new_son_list, new_father_list, empty_prob_list)
        retFactor.prob_list = empty_prob_list
        for var, val in observed_dict.items():
            observeFactor(retFactor, var, val)
        return retFactor


# EvidenceList: list of dictionary, ie: [a:true,b:False,c:False]
def resultFactor(FactorList, queryVariables, orderedListOfHiddenVariables, evidenceDict):
    # Set the observed variables to their observed values
    for factor in FactorList:
        for var, val in evidenceDict.items():
            if var in factor.variable_list:
                print("observe", var)
                observeFactor(factor, var, val)
    # Pick a hidden variable H, Join all factors mentioning H, Eliminate (sum out) H
    for hid_var in orderedListOfHiddenVariables:
        print("join:", hid_var)
        list_of_product = []
        for factor in FactorList:
            if hid_var in factor.variable_list:
                list_of_product.append(factor)
        for factor in list_of_product:
            FactorList.remove(factor)
        joined_factor = list_of_product[0]
        for i in range(1, len(list_of_product)):
            joined_factor = productFactor(joined_factor, list_of_product[i])
        sumFactor(joined_factor, hid_var)
        FactorList = [joined_factor] + FactorList
    # Join all remaining factors and normalize
    print("join remain: ", len(FactorList))
    joined_factor = FactorList[0]
    for i in range(1, len(FactorList)):
        joined_factor = productFactor(joined_factor, FactorList[i])
    normalizedFactor(joined_factor)
    print("final result:")
    joined_factor.print_factor()
