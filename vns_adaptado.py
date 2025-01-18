import random
import pprint
import copy
import numpy as np

class Depot:
    def __init__(self, x, y, capacity, index):
        self.x = x
        self.y = y
        self.capacity = capacity
        self.index = index

    def removeCapacity(self, value):
        self.capacity -= value


class Customer:
    def __init__(self, x, y, demand, index):
        self.x = x
        self.y = y
        self.index = index
        self.demand = demand

class Vehicle:
    def __init__(self, capacity, cost, fixed_cost, index):
        self.capacity = capacity
        self.filled = 0
        self.cost = cost
        self.fixed_cost = fixed_cost
        self.id = index

    def fill(self, value):
        self.filled = value

    def deliver(self, value):
        self.filled -= value

    def empty(self):
        self.filled = 0

    def remainingCapacity(self):
        return self.filled

    def __str__(self):
        str = "Vehicle: {} ".format(hex(id(self)))
        str += "capacity:{}".format(self.capacity)
        return str

class Route:
    def __init__(self, vehicle, start):
        self.vehicle = vehicle
        self.depot_index = start #depot index
        self.customer_visited = []
        self.total_cost = self.vehicle.fixed_cost
        self.total_delivered = 0
        self.total_distance = 0

    def addCost(self, cost):
        self.total_cost +=cost

    def addCustomer(self, customerIndex):
        self.customer_visited.append(customerIndex)

    def addDistance(self, distance):
        self.total_distance += distance

    def deliver(self, value):
        self.total_delivered +=  value
        self.vehicle.deliver(value)

    def add(self, customerIndex, value, distance):
        self.addCost(self.vehicle.cost*distance)
        self.addCustomer(customerIndex)
        self.deliver(value)
        self.addDistance(distance)

    def lastVisited(self):
        if len(self.customer_visited) > 0:
            return self.customer_visited[-1]
        else:
            return -1

    def getDepotIndex(self):
        return self.depot_index

    def __str__(self):
        str = "Route: "
        str += "cost: {} \n".format(self.total_cost)
        str += "depot index: {} \n".format(self.depot_index)
        str += "delivered: {} \n".format(self.total_delivered)
        str += "route: {} ".format(self.customer_visited)
        return str

    def __repr__(self):
        return self.__str__()

def generate_sample_data():
    depots = [
        Depot(10, 10, 100,0),
        Depot(20, 20, 120,1),
        Depot(30, 30, 150,2)
    ]

    customers = [
        Customer(12, 15, 10,0),
        Customer(25, 18, 15,1),
        Customer(18, 24, 20,2),
        Customer(5, 5, 5,3),
        Customer(27, 30, 25,4),
        Customer(13, 16, 12,5),
        Customer(23, 26, 22,6),
        Customer(9, 8, 7,7)
    ]

    vehicles = [
        Vehicle(50, 10, 300, 0),
        Vehicle(60, 12, 200, 1),
        Vehicle(80, 15, 150, 2)
    ]

    distance_matrix = generate_distance_matrix(depots, customers)
    return depots, customers, vehicles, distance_matrix


def generate_distance_matrix(depots, customers):
    """Generate a distance matrix between all depots and customers."""
    points = depots + customers
    n = len(points)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = ((points[i].x - points[j].x) ** 2 + (points[i].y - points[j].y) ** 2) ** 0.5
    return matrix


class MultiDepotVRP:
    def __init__(self, depots, customers, vehicles, distance_matrix, max_iterations, alpha):
        self.depots = depots  # List of depots with location and capacity
        self.customers = customers  # List of customer demands and locations
        self.vehicles = vehicles  # List of vehicles with capacity and cost
        self.distance_matrix = distance_matrix
        self.max_iterations = max_iterations
        self.alpha = alpha
        # pp = pprint.PrettyPrinter()
        # pp.pprint(distance_matrix)

    def createLRC(self, depot_index, customer_index, depot_list_size, customers, capacity):
        lrc = {}
        max_value = 0

        # prevent missint customer on final result
        if len(customers) <= 2:
            for customer in customers:
                distance = 0

                if depot_index > 0:
                    distance = self.distance_matrix[depot_index][depot_list_size + customer.index]
                else:
                    distance = self.distance_matrix[depot_list_size + 
                               customer_index][depot_list_size + customer.index]

                lrc[customer] = distance

            return lrc

        for customer in customers:
            distance = 0

            if not customer_index > 0:
                distance = self.distance_matrix[depot_index][depot_list_size + customer.index]
            else:
                distance = self.distance_matrix[depot_list_size + customer_index][depot_list_size + customer.index]

            if max_value < distance:
                max_value = distance

        max_value = max_value * self.alpha
        
        for customer in customers:
            distance = 0

            if depot_index > 0:
                distance = self.distance_matrix[depot_index][depot_list_size + customer.index]
            else:
                distance = self.distance_matrix[depot_list_size + 
                           customer_index][depot_list_size + customer.index]

            if distance <= max_value and customer.demand <= capacity:
                lrc[customer] = distance

        return lrc


    def returnToDepot(self, route, depot_list_size):
        last_visited = route.lastVisited()
        if(last_visited != -1):
            distance = self.distance_matrix[route.getDepotIndex()][depot_list_size + last_visited]

            route.addCost(route.vehicle.cost * distance)
            route.addDistance(distance)

    def getSolutionValues(self, solution):
        total_distance = 0
        total_cost = 0
        total_delivered = 0
        total_customers_served = 0


        if solution:
            for route in solution:
                total_distance += route.total_distance
                total_cost += route.total_cost
                total_delivered += route.total_delivered
                total_customers_served += len(route.customer_visited)

        return total_distance, total_cost, total_delivered, total_customers_served


    def generate_initial_solution(self, vehicles, depots, customers):
        """Create an initial naive solution (assign each customer to the nearest depot)."""
        solution = []
        
        depot_list_size = len(depots)

        while (len(vehicles) > 0 and 
              len(customers) > 0):

            vehicle = random.choice(vehicles)
            depot = random.choice(depots)

            vehicle.fill(min(vehicle.capacity, depot.capacity))
            depot.removeCapacity(vehicle.remainingCapacity())

            vehicles.remove(vehicle)

            if depot.capacity <= 0:
                depots.remove(depot)

            route = Route(vehicle, depot.index)
            solution.append(route)

            customer_index = 0

            while len(customers) > 0 and vehicle.remainingCapacity() > 0:

                lrc = self.createLRC(depot.index, customer_index, depot_list_size, customers, 
                                vehicle.remainingCapacity())

                if len(lrc) > 0:
                    customer_in_dict = random.choice(list(lrc.keys()))

                    route.add(customer_in_dict.index, 
                              customer_in_dict.demand, lrc[customer_in_dict])

                    customers.remove(customer_in_dict)
                    customer_index = customer_in_dict.index

                else:
                    self.returnToDepot(route, depot_list_size)
                    break

        return solution

    def recalculateRoute(self, depot_index, customers_list):

        distance = 0
        depot = depot_index 
        previous_index = -1
        depot_list_size = len(self.depots)

        for index in customers_list:

            if depot > 0:
                distance = self.distance_matrix[depot][depot_list_size + index]
                depot = 0
            else:
                distance = self.distance_matrix[depot_list_size + 
                           previous_index][depot_list_size + index]

            previous_index = index

        distance += self.distance_matrix[depot_index][depot_list_size + previous_index]

        return distance



    def two_opt_swap(self, solution, i, j, route):
        """Perform a 2-opt swap to try and improve the solution."""
        # Swap elements between positions i and j in the same route

        if len(solution) < 2:
            return False  # Skip routes that are too small to optimize

        # Check if i and j are valid within the route
        if i < len(solution) and j < len(solution) and i < j:
            # Perform a 2-opt swap on the route
            new_route = solution[:i] + solution[i:j+1][::-1] + solution[j+1:]

            # Calculate the cost difference and keep the swap if it's beneficial
            old_cost = route.total_distance
            new_cost = self.recalculateRoute(route.depot_index, new_route)
            if new_cost < old_cost:
                route.customer_visited = copy.deepcopy(new_route)
                route.total_cost = route.vehicle.cost*new_cost
                route.total_distance = new_cost
                solution = new_route
                return True
        return False

    def search(self, solution):

        for route in solution:
            improved = True
            while improved:
                improved = False
                for i in range(len(route.customer_visited)):
                    for j in range(i + 1, len(route.customer_visited)):
                        if self.two_opt_swap(route.customer_visited, i, j, route):
                            improved = True
        return solution


    def grasp(self):
        best_solution = []

        for _ in range(self.max_iterations):
            initial_solution = self.generate_initial_solution(copy.deepcopy(vehicles), 
                                                         copy.deepcopy(depots), 
                                                         copy.deepcopy(customers))
            # print(self.getSolutionValues(initial_solution))
            # return initial_solution
            # print("\n\n")
            # print(initial_solution)
            # print(self.getSolutionValues(initial_solution))
            final_solution = self.search(initial_solution)

            (best_distance, 
             best_cost, 
             best_delivered,
             best_customers_served) = self.getSolutionValues(best_solution)

            (final_distance, 
             final_cost, 
             final_delivered,
             final_customers_served) = self.getSolutionValues(final_solution)
            # print(final_solution)
            # print(self.getSolutionValues(final_solution))
            # print("\n\n")

            # print(best_distance, 
            #              best_cost, 
            #              best_delivered,
            #              best_customers_served)

            # print(final_distance, 
            #  final_cost, 
            #  final_delivered,
            #  final_customers_served)

            if best_customers_served < final_customers_served:
                best_solution = copy.deepcopy(final_solution)
            elif (best_delivered <= final_delivered and 
                best_cost >= final_cost):
                best_solution = copy.deepcopy(final_solution)

        print(self.getSolutionValues(best_solution))
        return best_solution

    def find_best_solution(self, routes):

        best_route = routes[0]

        for route in routes[1:]:

            if len(route.customer_visited) > len(best_route.customer_visited):
                best_route = route
            elif (route.total_delivered >= best_route.total_delivered and 
                 route.total_cost <= best_route.total_cost):
                 best_route = route
        return best_route


    def generate_neighborhood(self, initial_solution, vehicles, depots, customers ):

        best_solution = self.find_best_solution(initial_solution)

        # print(best_solution)
        # print(vehicles)
        vehicles.remove(vehicles[best_solution.vehicle.id])
        depots.remove(depots[best_solution.depot_index])

        for customer in customers:
            if customer.index in best_solution.customer_visited:
                customers.remove(customer)



        new_solutions = [best_solution]

        depot_list_size = len(depots)

        while (len(vehicles) > 0 and 
              len(customers) > 0):

            vehicle = random.choice(vehicles)
            depot = random.choice(depots)

            vehicle.fill(min(vehicle.capacity, depot.capacity))
            depot.removeCapacity(vehicle.remainingCapacity())

            vehicles.remove(vehicle)

            if depot.capacity <= 0:
                depots.remove(depot)

            route = Route(vehicle, depot.index)
            new_solutions.append(route)

            customer_index = 0

            while len(customers) > 0 and vehicle.remainingCapacity() > 0:

                lrc = self.createLRC(depot.index, customer_index, depot_list_size, customers, 
                                vehicle.remainingCapacity())

                if len(lrc) > 0:
                    customer_in_dict = random.choice(list(lrc.keys()))

                    route.add(customer_in_dict.index, 
                              customer_in_dict.demand, lrc[customer_in_dict])

                    customers.remove(customer_in_dict)
                    customer_index = customer_in_dict.index

                else:
                    self.returnToDepot(route, depot_list_size)
                    break

        return new_solutions



    def vns(self):

        best_solution = []

        initial_solution = self.generate_initial_solution(copy.deepcopy(self.vehicles), 
                                                         copy.deepcopy(self.depots), 
                                                         copy.deepcopy(self.customers))
        best_solution = initial_solution

        for _ in range(self.max_iterations):

            new_solution = self.generate_neighborhood(copy.deepcopy(best_solution), 
                                                      copy.deepcopy(self.vehicles), 
                                                      copy.deepcopy(self.depots), 
                                                      copy.deepcopy(self.customers))

            final_solution = self.search(initial_solution)
            

            (best_distance, 
             best_cost, 
             best_delivered,
             best_customers_served) = self.getSolutionValues(best_solution)

            (new_distance, 
             new_cost, 
             new_delivered,
             new_customers_served) = self.getSolutionValues(final_solution)
            
            if best_customers_served < new_customers_served:
                best_solution = copy.deepcopy(new_solution)
            elif (best_customers_served == new_customers_served and 
                  best_cost > new_cost):
                best_solution = copy.deepcopy(new_solution)
            elif (best_delivered <= new_delivered and 
                best_cost >= new_cost):
                best_solution = copy.deepcopy(new_solution)

        print(self.getSolutionValues(best_solution))
        return best_solution


    def simulated_annealing(self, route, max_iter=1000, temp_inicial=1000, alfa=0.99):
        """Implementa a meta-heurística Simulated Annealing."""
        best_solution = route
        
        (best_distance, 
             best_cost, 
             best_delivered,
             best_customers_served) = self.getSolutionValues(best_solution)

        temperature = temp_inicial

        for _ in range(max_iter):
            new_solution = self.generate_neighborhood(copy.deepcopy(best_solution), 
                                                      copy.deepcopy(self.vehicles), 
                                                      copy.deepcopy(self.depots), 
                                                      copy.deepcopy(self.customers))

            final_solution = self.search(new_solution)

            (distance, 
             cost, 
             delivered,
             customers_served) = self.getSolutionValues(final_solution)


            delta = cost - best_cost
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                if cost < best_cost:
                    best_solution = copy.deepcopy(final_solution)
                    best_cost = cost

            temperature *= alfa

        print("\n\n\n")
        print(self.getSolutionValues(best_solution))
        return best_solution


def read_instance(filename):
    with open(filename, 'r') as file:
        num_depots = int(file.readline().strip())
        num_customers = int(file.readline().strip())
        num_vehicles = int(file.readline().strip())
        
        depots = []
        customers = []
        vehicles = []
        
        # Lendo os depósitos
        for depot_index in range(num_depots):
            x, y, capacity = map(int, file.readline().strip().split())
            depots.append(Depot(x, y, capacity, depot_index))
        
        # Lendo os clientes
        for customer_index in range(num_customers):
            x, y, demand = map(int, file.readline().strip().split())
            customers.append(Customer(x, y, demand, customer_index))

        for vehicle_index in range(num_vehicles):
            capacity, fixed_cost, cost_per_distance = map(int, file.readline().strip().split())
            vehicles.append(Vehicle(capacity, cost_per_distance, fixed_cost, vehicle_index))

        distance_matrix = generate_distance_matrix(depots, customers)
        
        return depots, customers, vehicles, distance_matrix


max_iterations = 1000
alpha = 1
#depots, customers, vehicles, distance_matrix = generate_sample_data()
depots, customers, vehicles, distance_matrix = read_instance("mdvrp_input.txt")
problem = MultiDepotVRP(depots, customers, vehicles, distance_matrix, max_iterations, alpha)
solution = problem.vns()
print(solution)
print(problem.simulated_annealing(solution))