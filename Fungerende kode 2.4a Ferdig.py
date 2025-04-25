#Importing the different libraries needed for the code


import pyomo.environ as pyo
from pyomo.opt import SolverFactory


#B-matrix

B_matrix = [[-30, 20, 10],
            [20, -50, 30],
            [10, 30, -40]]

#Defining the different lines and where they start and end

Line_from = {1:1, 2:1, 3:2 }
Line_to = {1:2, 2:3, 3:3}

#Dictionaries over all known values

Generator_cap = {1: 300, 2: 400, 3: 300, 4: 1000, 5: 1000}
Demand = {1: 200, 2: 0, 3: 500} #Demand at node 2 is split below
Line_capacity = {1: 500, 2: 500, 3: 100}
Marginal_costs = {1: 200, 2: 300, 3: 800, 4: 1000, 5: 600}

#Defining which generators belong to which node

Generator_to_node = {1: 1, 2:1, 3:1, 4:2, 5:3}

#Split node 2

Load_node_2 = {1:200, 2: 250, 3:250}

#Making the function to run the OPF

def OPF():
    
    model = pyo.ConcreteModel() #Using pyomo and concrete model as the model in our optimization
    
    
    #Defining the sets, have added 2 new sets to cover all we need
    
    model.nodes = pyo.Set(ordered = True, initialize = [1,2,3])   
    model.generators = pyo.Set(ordered = True, initialize = [1,2,3,4,5])
    model.lines = pyo.Set(ordered = True, initialize = [1,2,3])
    
    
    #Defining the parameters
    
    model.demand = pyo.Param(model.nodes, initialize = Demand)  #Demand
    
    model.gen_cap = pyo.Param(model.generators, initialize = Generator_cap) #Max power generation
    
    model.mc = pyo.Param(model.generators, initialize = Marginal_costs) #Marginal costs
    
    model.pu_base = pyo.Param(initialize = 1000) #PU base
    
    model.line_cap = pyo.Param(model.lines, initialize = Line_capacity) #Line capacity
    
    model.line_from   = pyo.Param(model.lines, initialize = Line_from ) #Parameter for starting node for every line
    
    model.line_to     = pyo.Param(model.lines, initialize = Line_to) #Parameter for ending node for every line
    
    
    #Defining the variables

    model.theta = pyo.Var(model.nodes) #Voltage angles
     
    model.gen = pyo.Var(model.generators) #Power generated                                    
     
    model.flow = pyo.Var(model.lines) #Power flow

                                          
    #Defining the objective function
    
    def ObjRule(model): #Minimize the cost of generation
        return (sum(model.gen[n]*model.mc[n] for n in model.generators))
    model.OBJ = pyo.Objective(rule = ObjRule, sense = pyo.minimize)  
    
    
    #Defining the constraints
    
    def Max_gen(model,n): #Setting the maximum generation
        return(model.gen[n] <= model.gen_cap[n])
    model.Max_gen_const = pyo.Constraint(model.generators, rule = Max_gen)
    
    
    def Min_gen(model, n): #Setting the minimum generation (can't generate less then zero)
        return model.gen[n] >= 0
    model.Min_gen_const = pyo.Constraint(model.generators, rule=Min_gen)
    
    
    def FlowBalDC_max(model,n): #Max load flow
        return(model.flow[n] <= model.line_cap[n])
    model.FlowBalDC_max_const = pyo.Constraint(model.lines, rule = FlowBalDC_max)
    
    
    def FlowBalDC_min(model,n): #Min load flow (Can't transfer more then the maximum value in negative)
        return(model.flow[n] >= -model.line_cap[n])
    model.FlowBalDC_min_const = pyo.Constraint(model.lines, rule = FlowBalDC_min)


    def ref_node(model): #Setting node 1 as the reference node
        return(model.theta[1] == 0)
    model.ref_node_const = pyo.Constraint(rule = ref_node)
    
    
    #Load balance, that generation meets demand and transfer
    #The constraint has been changed to take into consideration that node 2 has 3 loads
    
    def LoadBal(model,n):
        if n == 2 :
            demand = sum(Load_node_2.values()) 
            return (sum(model.gen[g] for g in model.generators if Generator_to_node[g] == n)) == demand +\
            sum(B_matrix[n-1][o-1]*model.theta[o]*model.pu_base for o in model.nodes)
        else :
            return (sum(model.gen[g] for g in model.generators if Generator_to_node[g] == n)) == model.demand[n] +\
            sum(B_matrix[n-1][o-1]*model.theta[o]*model.pu_base for o in model.nodes)
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


    #Print the results we need    

    print()
    print("--- DCOPF Results ---")
    for g in model.generators: #Calculate the values and costs for the different generators
        gen_val = pyo.value(model.gen[g])
        cost = gen_val * pyo.value(model.mc[g])
        print()
        print(f"Generator {g}:")
        print(f"Generation = {gen_val:.2f} MW")
        print(f"Cost = {cost:.2f} NOK")
    print()
    print("Bus Angles:")
    for n in model.nodes:
        print(f"Theta[{n}] = {pyo.value(model.theta[n]):.4f} rad")
    print()
    print("Line Flows:")
    for l in model.lines:
        print(f"Line {l} ({model.line_from[l]} → {model.line_to[l]}): Flow = {pyo.value(model.flow[l]):.2f} MW")
    print()
    print(f"Total Cost: {pyo.value(model.OBJ):.2f} NOK")
    print()
    print("Nodal Prices (dual values):")
    for n in model.nodes :
        print(f"Node {n} {model.dual[model.LoadBal_const[n]]:.2f} NOK/MWh")
    print()
    print("Line Prices (dual values):")
    for n in model.lines :
        print(f"line {n} {model.dual[model.FlowBal_const[n]]/model.pu_base:.2f} NOK/MWh")
    print()
    print("Generator Prices (dual values):")
    for n in model.generators :
        print(f"Generator {n} {model.dual[model.Max_gen_const[n]]:.2f} NOK/MWh")
    
    return()



OPF()