#Importing the different libraries needed for the code

import pyomo.environ as pyo
from pyomo.opt import SolverFactory


#B-matrix

B_matrix = [[-30, 20, 10],
            [20, -50, 30],
            [10, 30, -40]]

#Defining the different lines and where they start and end

Line_from = {1:1, 2:1, 3:2}
Line_to = {1:2, 2:3, 3:3}

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
    
    model.line_from   = pyo.Param(model.set, initialize = Line_from ) #Parameter for starting node for every line
    
    model.line_to     = pyo.Param(model.set, initialize = Line_to)  #Parameter for ending node for every line
    
    
    #Defining the variables

    model.theta = pyo.Var(model.set) #Voltage angles
     
    model.gen = pyo.Var(model.set) #Power generated                                    
     
    model.flow = pyo.Var(model.set) #Power flow

                                          
    #Defining the objective function
    
    def ObjRule(model): #Minimize the cost of generation
        return (sum(model.gen[n]*model.mc[n] for n in model.set))
    model.OBJ = pyo.Objective(rule = ObjRule, sense = pyo.minimize)  
    
    
    #Defining the constraints
    
    def Max_gen(model,n): #Setting the maximum generation
        return(model.gen[n] <= model.gen_cap[n])
    model.Max_gen_const = pyo.Constraint(model.set, rule = Max_gen)
    
    def Min_gen(model, n): #Setting the minimum generation (can't generate less then zero)
        return model.gen[n] >= 0
    model.Min_gen_const = pyo.Constraint(model.set, rule=Min_gen)
    
    def FlowBalDC_max(model,n): #Max load flow
        return(model.flow[n] <= model.line_cap[n])
    model.FlowBalDC_max_const = pyo.Constraint(model.set, rule = FlowBalDC_max)
    
    def FlowBalDC_min(model,n): #Min load flow (Can't transfer more then the maximum value in negative)
        return(model.flow[n] >= -model.line_cap[n])
    model.FlowBalDC_min_const = pyo.Constraint(model.set, rule = FlowBalDC_min)

    def ref_node(model): #Setting node 1 as the reference node
        return(model.theta[1] == 0)
    model.ref_node_const = pyo.Constraint(rule = ref_node)
    
    
    #Load balance, that generation meets demand and transfer
    
    def LoadBal(model,n): 
        return(model.gen[n] == model.demand[n] +\
        sum(B_matrix[n-1][o-1]*model.theta[o]*model.pu_base for o in model.set))
    model.LoadBal_const = pyo.Constraint(model.set, rule = LoadBal)
    
    
    #Flow balance; that flow in line is equal to change in phase angle multiplied with the admittance for the line
    
    def FlowBal(model,l):
        return(model.flow[l]/model.pu_base == ((model.theta[model.line_from[l]]- model.theta[model.line_to[l]])*-B_matrix[model.line_from[l]-1][model.line_to[l]-1]))
    model.FlowBal_const = pyo.Constraint(model.set, rule = FlowBal)
    
    
    #Compute the optimization problem

    #Set the solver for this
    opt         = SolverFactory('gurobi',solver_io="python")
    
    #Enable dual variable reading -> important for dual values of results
    model.dual  = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    #Solve the problem
    results     = opt.solve(model, load_solutions = True)
    
    #Write result on performance
    results.write(num=1)


    #Print the results we need
    
    print()
    print("--- DCOPF Results ---")
    for n in model.set:
        print()
        print(f"Bus {n}:") 
        print(f"Generation = {model.gen[n].value:.2f} MW")
        print(f"Theta = {model.theta[n].value:.4f} rad")
        print(f"Demand = {model.demand[n]:.2f} MW")
    print()
    print(f"Total Generation Cost = {model.OBJ():.2f} NOK")
    print()
    print("Nodal Prices (dual values):")
    for n in model.set :
        print(f"Node {n} {model.dual[model.LoadBal_const[n]]:.2f} NOK/MWh")
    print()
    print("Line Prices (dual values):")
    for n in model.set :
        print(f"Line {n} {model.dual[model.FlowBal_const[n]]/model.pu_base:.2f} NOK/MWh") 
    print()
    print("Line Flows:")
    for l in model.set:
        print(f"Line {l} ({model.line_from[l]} → {model.line_to[l]}): Flow = {pyo.value(model.flow[l]):.2f} MW")
    
    return()

#Run the function

OPF()