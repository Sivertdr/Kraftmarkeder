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

#Marginal Cost for the different demands in node 2

Marginal_costs_demand = {1: 1300 , 2: 800 , 3: 500}

#The marginal costs for node 2 set in a dictonary

Load_data = {
    1: {"node" : 2, "demand": 200, "WTP": 1300},
    2: {"node" : 2, "demand": 250, "WTP": 800},
    3: {"node" : 2, "demand": 250, "WTP": 500},
    }

Clean_generators = [4] #The generator that is clean

CO2_limit = 950000 #The total emission  from task 2.5b

emissions = {1: 1500 , 2: 700 , 3: 100 , 4: 0 , 5: 1000} #Emissions from the different generators

#Making the function to run the OPF

def OPF():
    
    model = pyo.ConcreteModel() #Using pyomo and concrete model as the model in our optimization
    
    
    #Defining the sets
    
    model.nodes = pyo.Set(ordered = True, initialize = [1,2,3])   
    model.generators = pyo.Set(ordered = True, initialize = [1,2,3,4,5])
    model.lines = pyo.Set(ordered = True, initialize = [1,2,3])
    model.loads = pyo.Set(ordered = True, initialize = Load_data.keys())
    
    #Defining the parameters
    
    model.demand = pyo.Param(model.nodes, initialize = Demand)  #Demand
    
    model.gen_cap = pyo.Param(model.generators, initialize = Generator_cap) #Max power generation
    
    model.mc = pyo.Param(model.generators, initialize = Marginal_costs) #Marginal costs
    
    model.pu_base = pyo.Param(initialize = 1000) #PU base
    
    model.line_cap = pyo.Param(model.lines, initialize = Line_capacity) #Line capacity
    
    model.line_from   = pyo.Param(model.lines, initialize = Line_from )         #Parameter for starting node for every line
    
    model.line_to     = pyo.Param(model.lines, initialize = Line_to)           #Parameter for ending node for every line

    #Fetching the data from the Load_data with data regarding to WTP
    
    model.load_node = pyo.Param(model.loads, initialize={l : Load_data[l]["node"] for l in Load_data})
    
    model.load_demand = pyo.Param(model.loads, initialize={l : Load_data[l]["demand"] for l in Load_data})
    
    model.load_WTP = pyo.Param(model.loads, initialize={l : Load_data[l]["WTP"] if Load_data[l]["WTP"] else 0 for l in Load_data})
    
    #Defining the CO_emissions as a parameter 
    
    model.CO2_emissions = pyo.Param(model.generators, initialize = emissions)
    

    #Defining the variables

    model.theta = pyo.Var(model.nodes) #Voltage angles
     
    model.gen = pyo.Var(model.generators) #Power generated                                    
     
    model.flow = pyo.Var(model.lines) #Power flow
    
    model.load_supplied = pyo.Var(model.nodes, within=pyo.NonNegativeReals) #How much of the load is supplied
      
    #Defining a new objective function                                

    def ObjRule(model): #The objective function now wants to maximize social welfare
        return (sum(model.load_supplied[n] * model.load_WTP[n] for n in model.loads)) - (sum(model.gen[g] * model.mc[g] for g in model.generators))
    model.OBJ = pyo.Objective(rule = ObjRule, sense = pyo.maximize)
    
    
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
    
    
    def supply_limit(model, n): #Supplied power can not exceed the demand
        return(model.load_supplied[n] <= model.load_demand[n])
    model.supply_limit = pyo.Constraint(model.loads, rule = supply_limit)
     
    
    def cap_and_trade(model): #Rule for cap and trade, the total emissions can not exceed the limit
        return(sum(model.CO2_emissions[p] * model.gen[p] for p in model.generators) <= CO2_limit)
    model.CO2_constraint = pyo.Constraint(rule = cap_and_trade)
    
    
    #Load balance; that generation meets demand, shedding, and transfer from lines and cables
    #Same as in 2.4b
    
    def LoadBal(model,n):
        gen = sum(model.gen[g] for g in model.generators if Generator_to_node[g] == n)
        demand = sum(model.load_supplied[l] for l in model.loads if model.load_node[l] == n)
        if n in model.demand:
            demand += model.demand[n]
        transfer = sum(B_matrix[n-1][o-1] * model.theta[o] * model.pu_base for o in model.nodes)
        return gen == demand + transfer
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
    print("Supplied Loads at Node 2:")
    for n in model.nodes:
        print(f" Load {n}: Supplied {pyo.value(model.load_supplied[n]):.2f} / {model.load_demand[n]} MW ")
    print()
    print(f"Total Cost: {pyo.value(model.OBJ):.2f} NOK")
    print()
    print("Nodal Prices (dual values):")
    for n in model.nodes :
        print(f" Node {n} {model.dual[model.LoadBal_const[n]]:.2f} NOK/MWh")
    print()
    print("Line Prices (dual values):")
    for n in model.lines :
        print(f" Node {n} {model.dual[model.FlowBal_const[n]]/model.pu_base:.2f} NOK/MWh")
    print()
    print("Generator Prices (dual values):")
    for n in model.generators :
        print(f" Generator {n} {model.dual[model.Max_gen_const[n]]:.2f} NOK/MWh")
    print()
    print("Carbon tax (dual values):")
    print(f"Carbon tax {model.dual[model.CO2_constraint]:.2f} NOK/kg")
    
    return()



OPF()
