#Importing the different libraries needed for the code

import numpy as np
import sys
import time
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


#B-matrix

B_matrix = [[-30, 20, 10],
            [20, -50, 30],
            [10, 30, -40]]



#Dictionaries over all known values

Generator_cap = {1: 1000, 2: 1000, 3: 1000}
Demand = {1: 200, 2: 200, 3:500}
Line_capacity = {1: 500, 2: 500, 3: 100}
Marginal_costs = {1: 300, 2: 1000, 3: 600}



#Making the function to run the OPF

def OPF():
    
    model = pyo.ConcreteModel() #Using pyomo and concrete model as the model in our optimization
    
    
    #Defining the set
    
    model.set = pyo.Set(ordered = True, initialize = [1,2,3])   

    
    #Defining the parameters
    
    model.demand = pyo.Param(model.set, initialize = Demand)  #Demand
    
    model.gen_cap = pyo.Param(model.set, initialize = Generator_cap) #Max power generation
    
    model.mc = pyo.Param(model.set, initialize = Marginal_costs) #Marginal costs
    
    model.pu_base = pyo.Param(initialize = 1000) #PU base
    
    model.line_cap = pyo.Param(model.set, initialize = Line_capacity) #Line capacity
    
    
    #Defining the variables

    model.theta = pyo.Var(model.set) #Voltage angles
     
    model.gen = pyo.Var(model.set) #Power generated                                    
     
    model.flow = pyo.Var(model.set) #Power flow

                                          
    #Defining the objective function
    
    def ObjRule(model): 
        return (sum(model.gen[n]*model.mc[n] for n in model.set))
    model.OBJ = pyo.Objective(rule = ObjRule, sense = pyo.minimize)  
    
    
    #Defining the constraints
    
    def Max_gen(model,n): #Setting the maximum generation
        return(model.gen[n] <= model.gen_cap[n])
    model.Max_gen_const = pyo.Constraint(model.set, rule = Max_gen)
    
    
    def FlowBalDC_max(model,n): #Max load flow
        return(model.flow[n] <= model.line_cap[n])
    model.FlowBalDC_max_const = pyo.Constraint(model.set, rule = FlowBalDC_max)
    

    
    def FlowBalDC_min(model,n): #Min load flow
        return(model.flow[n] >= -model.line_cap[n])
    model.FlowBalDC_min_const = pyo.Constraint(model.set, rule = FlowBalDC_min)



    def ref_node(model):
        return(model.theta[1] == 0)
    model.ref_node_const = pyo.Constraint(rule = ref_node)
    
    
    #Load balance; that generation meets demand, shedding, and transfer from lines and cables
    
    
    def LoadBal(model,n):
        return(model.gen[n] == model.demand[n] +\
        sum(B_matrix[n-1][o-1]*model.theta[o]*model.pu_base for o in model.set))
    model.LoadBal_const = pyo.Constraint(model.set, rule = LoadBal)
    
    #Flow balance; that flow in line is equal to change in phase angle multiplied with the admittance for the line
    
#    def FlowBal(model,l):
#        return(model.flow[l]/model.Pu_base == ((model.theta[model.AC_from[l]]- model.theta[model.AC_to[l]])*-Data["B-matrix"][model.AC_from[l]-1][model.AC_to[l]-1]))
#    model.FlowBal_const = pyo.Constraint(model.L, rule = FlowBal)
    
    
    
    #Compute the optimization problem

    #Set the solver for this
    opt         = SolverFactory('gurobi',solver_io="python")
    

    #Enable dual variable reading -> important for dual values of results
    model.dual      = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    
    #Solve the problem
    results     = opt.solve(model, load_solutions = True)
    
    #Write result on performance
    results.write(num=1)

    #Run function that store results
    #Store_model_data(model)
    
    return()


#Function to store the data
"""
def Store_model_data(model):
    
    
    #Stores the results from the optimization model run into an excel file
    
    
    
    #Create empty dictionaries that will be filled
    NodeData    = {}
    DCData      = {}
    MiscData    = {}
    

    
    #Node data
    
    #Write dictionaries for each node related value
    Theta       = {}
    Gen         = {}
    Shed        = {}
    Demand      = {}
    CostGen     = {}
    CostShed    = {}
    DualNode    = {}    
    
    #For every node, store the data in the respective dictionary
    for node in model.set:
        
    
        Theta[node]         = round(model.theta[node].value,4)
        DualNode[node]      = round(model.dual[model.LoadBal_const[node]],1) 
        Gen[node]           = round(model.gen[node].value,4)
        Demand[node]        = round(model.demand[node],4)
        CostGen[node]       = round(model.gen[node].value*model.mc[node],4)
       
        
        
    
    
    #Store Node Data
    NodeData["Theta [rad]"] = Theta
    NodeData["Gen"]         = Gen
    NodeData["Shed"]        = Shed
    NodeData["Demand"]      = Demand
    NodeData["MargCost"]    = Marginal_costs
    NodeData["CostGen"]     = CostGen
    NodeData["CostShed"]    = CostShed
    NodeData["Node Name"]   = {1: "Node1", 2: "Node2", 3: "Node3"}
    NodeData["Price"]       = DualNode

        
    
    #DC-line data
    DCFlow      = {}
    
    #For every cable, store the result
    for cable in model.set:
        DCFlow[cable]       = round(model.flow_DC[cable].value,4)
    
    DCData["DC Flow"]           = DCFlow
    DCData["Capacity"]          = Generator_cap

    
    
    #Misc
    Objective   = round(model.OBJ(),4)
    DCOPF       = [1]    
    MiscData["Objective"]   = {1:Objective}
    MiscData["DCOPF"]       = {2:DCOPF}  
    
    
    #Convert the dictionaries to objects for Pandas
    NodeData    = pd.DataFrame(data=NodeData)
    DCData      = pd.DataFrame(data=DCData)
    MiscData    = pd.DataFrame(data=MiscData) 
    
    #Decide what the name of the output file should be
    if Data["DCFlow"] == True:
        output_file = "DCOPF_results.xlsx"
    else:
        output_file = "ATC_results.xlsx"
    
    #Store each result in an excel file, given a separate sheet
    with pd.ExcelWriter(output_file) as writer:
        NodeData.to_excel(writer, sheet_name= "Node")
        DCData.to_excel(writer, sheet_name= "DC")
        MiscData.to_excel(writer, sheet_name= "Misc")
        
    print("\n\n")
    print("The results are now stored in the excel file: " + output_file)
    print("This program will now end")

    return()
"""
OPF()