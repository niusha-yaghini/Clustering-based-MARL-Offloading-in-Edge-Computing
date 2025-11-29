import pandas as pd
import numpy as np

class MEC:
    def __init__(self, id, private_cpu_capacity, public_cpu_capacity, memory_capacity):
        self.id = id
        self.private_cpu_capacity = private_cpu_capacity  # Private CPU processing capacity
        self.public_cpu_capacity = public_cpu_capacity  # Public CPU processing capacity
        self.memory_capacity = memory_capacity  # Memory capacity
        self.private_queue = []  # Private queue for local tasks
        self.public_queue = []  # Public queue for offloaded tasks

    def get_info(self):
        return {
            'id': self.id,
            'private_cpu_capacity': self.private_cpu_capacity,
            'public_cpu_capacity': self.public_cpu_capacity,
            'memory_capacity': self.memory_capacity
        }

class Cloud:
    def __init__(self, computational_capacity, memory_capacity):
        self.computational_capacity = computational_capacity
        self.memory_capacity = memory_capacity  # Memory capacity of the Cloud

    def get_info(self):
        return {
            'computational_capacity': self.computational_capacity,
            'memory_capacity': self.memory_capacity
        }

class Environment:
    def __init__(self, num_servers, cloud_capacity, cloud_memory, server_cpu_capacities, server_memory_capacities):
        self.num_servers = num_servers
        self.cloud = Cloud(cloud_capacity, cloud_memory)  # Cloud with computational capacity and memory
        self.servers = [MEC(i, server_cpu_capacities[i], 
                           server_cpu_capacities[i],  # public_cpu_capacity is same as private in this example
                           server_memory_capacities[i]) 
                        for i in range(num_servers)]

    def save_environment_csv(self, filename='environment.csv'):
        # Collecting server and cloud info for CSV
        server_data = []
        for server in self.servers:
            server_info = server.get_info()
            server_data.append({
                'Server ID': server_info['id'],
                'Private CPU Capacity': server_info['private_cpu_capacity'],
                'Public CPU Capacity': server_info['public_cpu_capacity'],
                'Memory Capacity': server_info['memory_capacity']
            })
        
        # Saving server data into CSV
        df_servers = pd.DataFrame(server_data)
        df_servers.to_csv(filename, index=False)
        
        # Saving cloud data into CSV
        cloud_info = self.cloud.get_info()
        df_cloud = pd.DataFrame([cloud_info])
        df_cloud.to_csv('cloud_info.csv', index=False)

    def load_environment(self, filename='environment.csv'):
        # Loading server data from CSV
        df_servers = pd.read_csv(filename)
        server_data = df_servers.to_dict(orient='records')
        
        # Re-initializing servers with loaded data
        self.servers = [MEC(server['Server ID'], server['Private CPU Capacity'], 
                            server['Public CPU Capacity'], server['Memory Capacity']) 
                        for i, server in enumerate(server_data)]

    def get_environment_info(self):
        # Returning environment info including cloud and server data
        environment_info = {
            'cloud': self.cloud.get_info(),
            'servers': [server.get_info() for server in self.servers]
        }
        return environment_info

# Example Usage
server_cpu_capacities = [0.5] * 20  # According to the paper, each MEC has 0.5 processing capacity
server_memory_capacities = [2.0] * 20  # According to the paper, each MEC has 2.0 memory capacity
cloud_capacity = 3.0  # Cloud computational capacity, as mentioned in the paper
cloud_memory = 10.0  # Cloud memory capacity, as mentioned in the paper

# Initialize environment
env = Environment(num_servers=20, cloud_capacity=cloud_capacity, cloud_memory=cloud_memory,
                  server_cpu_capacities=server_cpu_capacities, server_memory_capacities=server_memory_capacities)

# Save environment to CSV
env.save_environment_csv()

# Load environment from CSV (for testing)
env2 = Environment(num_servers=20, cloud_capacity=cloud_capacity, cloud_memory=cloud_memory,
                   server_cpu_capacities=server_cpu_capacities, server_memory_capacities=server_memory_capacities)
env2.load_environment()

# Get environment info
env_info = env2.get_environment_info()
print(env_info)
