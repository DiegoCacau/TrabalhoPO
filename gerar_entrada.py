import random

def generate_input_file(file_path, num_customers, num_depots, num_vehicle, 
                        demand_range, capacity_depot_range, capacity_vehicle_range, 
                        cost_range, running_cost_range, min_max_x_y):
    with open(file_path, 'w') as file:
        file.write(f"{num_depots}\n")
        file.write(f"{num_customers}\n")
        file.write(f"{num_vehicle}\n")
        
        for _ in range(num_depots):
            file.write(f"{random.randint(*min_max_x_y)} {random.randint(*min_max_x_y)} {random.randint(*capacity_depot_range)}\n")

        for _ in range(num_customers):
            file.write(f"{random.randint(*min_max_x_y)} {random.randint(*min_max_x_y)} {random.randint(*demand_range)}\n")

        for _ in range(num_vehicle):
            file.write(f"{random.randint(*capacity_vehicle_range)} {random.randint(*cost_range)} {random.randint(*running_cost_range)}\n")


if __name__ == '__main__':
    file_path = 'mdvrp_input.txt'
    num_customers = 50
    num_depots = 10
    num_vehicle = 15
    demand_range = (1, 100) 
    capacity_depot_range = (40, 300)
    capacity_vehicle_range = (50, 150) 
    cost_range = (200, 500)  
    running_cost_range = (5, 20)
    min_max_x_y = (0, 300) 

    generate_input_file(file_path, num_customers, num_depots, 
                        num_vehicle, demand_range, capacity_depot_range, 
                        capacity_vehicle_range, cost_range, running_cost_range, min_max_x_y)
    print(f"Arquivo de entrada gerado em: {file_path}")
