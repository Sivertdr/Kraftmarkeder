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

Line_from = {1:1, 2:1, 3:2 }
Line_to = {1:2, 2:3, 3:3}

#Dictionaries over all known values

Generator_cap = {1: 300, 2: 400, 3: 300, 4: 1000, 5: 1000}
Demand = {1: 200, 2: 200, 3:500}
Line_capacity = {1: 500, 2: 500, 3: 100}
Marginal_costs = {1: 200, 2: 300, 3: 800, 4: 1000, 5: 600}

Generator_to_node = {1: 1, 2:1, 3:1, 4:2, 5:3}

#Making the function to run the OPF

def OPF():
    
    model = pyo.ConcreteModel() #Using pyomo and concrete model as the model in our optimization
    
    
    #Defining the set
    
    model.nodes = pyo.Set(ordered = True, initialize = [1,2,3])   
    model.generators = pyo.Set(ordered = True, initialize = [1,2,3,4,5])
    model.lines = pyo.Set(ordered = True, initialize = [1,2,3])
    
    #Defining the parameters
    
    model.demand = pyo.Param(model.nodes, initialize = Demand)  #Demand
    
    model.gen_cap = pyo.Param(model.generators, initialize = Generator_cap) #Max power generation
    
    model.mc = pyo.Param(model.generators, initialize = Marginal_costs) #Marginal costs
    
    model.pu_base = pyo.Param(initialize = 1000) #PU base
    
    model.line_cap = pyo.Param(model.lines, initialize = Line_capacity) #Line capacity
    
    model.line_from   = pyo.Param(model.lines, initialize = Line_from )         #Parameter for starting node for every line
    
    model.line_to     = pyo.Param(model.lines, initialize = Line_to)           #Parameter for ending node for every line
    
    
    #Defining the variables

    model.theta = pyo.Var(model.nodes) #Voltage angles
     
    model.gen = pyo.Var(model.generators) #Power generated                                    
     
    model.flow = pyo.Var(model.lines) #Power flow

                                          
    #Defining the objective function
    
    def ObjRule(model): 
        return (sum(model.gen[n]*model.mc[n] for n in model.generators))
    model.OBJ = pyo.Objective(rule = ObjRule, sense = pyo.minimize)  
    
    
    #Defining the constraints
    
    def Max_gen(model,n): #Setting the maximum generation
        return(model.gen[n] <= model.gen_cap[n])
    model.Max_gen_const = pyo.Constraint(model.generators, rule = Max_gen)
    
    
    def Min_gen(model, n): #Setting the minimum generation
        return model.gen[n] >= 0
    model.Min_gen_const = pyo.Constraint(model.generators, rule=Min_gen)
    
    
    def FlowBalDC_max(model,n): #Max load flow
        return(model.flow[n] <= model.line_cap[n])
    model.FlowBalDC_max_const = pyo.Constraint(model.lines, rule = FlowBalDC_max)
    
    
    def FlowBalDC_min(model,n): #Min load flow
        return(model.flow[n] >= -model.line_cap[n])
    model.FlowBalDC_min_const = pyo.Constraint(model.lines, rule = FlowBalDC_min)


    def ref_node(model): #Setting the reference node
        return(model.theta[1] == 0)
    model.ref_node_const = pyo.Constraint(rule = ref_node)
    
    
    #Load balance; that generation meets demand, shedding, and transfer from lines and cables
    
    
    def LoadBal(model,n):
        return((sum(model.gen[g] for g in model.generators if Generator_to_node[g] == n)) == model.demand[n] +\
        sum(B_matrix[n-1][o-1]*model.theta[o]*model.pu_base for o in model.nodes))
    model.LoadBal_const = pyo.Constraint(model.nodes, rule = LoadBal)


    
    #Flow balance; that flow in line is equal to change in phase angle multiplied with the admittance for the line
    
    def FlowBal(model,l):
        return(model.flow[l]/model.pu_base == ((model.theta[model.line_from[l]]- model.theta[model.line_to[l]])*-B_matrix[model.line_from[l]-1][model.line_to[l]-1]))
    model.FlowBal_const = pyo.Constraint(model.lines, rule = FlowBal)
    
    
    
    #Compute the optimization problem

    #Set the solver for this
    opt         = SolverFactory('gurobi',solver_io="python")
    

    #Enable dual variable reading -> important for dual values of results
    model.dual      = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    
    #Solve the problem
    results     = opt.solve(model, load_solutions = True)
    
    #Write result on performance
    results.write(num=1)

    print("\n=== Solver Status ===")
    print(f"Status: {results.solver.status}")
    print(f"Termination Condition: {results.solver.termination_condition}")

    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        print("\n=== DCOPF Results ===")
        for g in model.generators:
            gen_val = pyo.value(model.gen[g])
            cost = gen_val * pyo.value(model.mc[g])
            print(f"Generator {g}: Generation = {gen_val:.2f} MW, Cost = {cost:.2f} NOK")
        print("\nBus Angles:")
        for n in model.nodes:
            print(f"Theta[{n}] = {pyo.value(model.theta[n]):.4f} rad")
        print("\nLine Flows:")
        for l in model.lines:
            print(f"Line {l} ({model.line_from[l]} → {model.line_to[l]}): Flow = {pyo.value(model.flow[l]):.2f} MW")
        print(f"\nTotal Cost: {pyo.value(model.OBJ):.2f} NOK")
        print("\nNodal Prices (dual values):")
        for n in model.nodes :
            print(f" Node {n} {model.dual[model.LoadBal_const[n]]:.2f} NOK/MWh")
    else:
        print("Solver did not find an optimal solution.")
    
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