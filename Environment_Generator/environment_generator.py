import pandas as pd
import numpy as np
import os
from collections import deque

class MEC:
    def __init__(self, id, private_cpu_capacity, public_cpu_capacity, num_servers):
        self.id = id
        self.private_cpu_capacity = private_cpu_capacity  # Private CPU processing capacity
        self.public_cpu_capacity = public_cpu_capacity  # Public CPU processing capacity
        self.private_queue = deque()  # Private queue for local tasks (FIFO)
        self.public_queues = {i: deque() for i in range(num_servers) if i != self.id}  # Public queues for tasks from other MECs (FIFO)

    def get_info(self):
        return {
            'id': self.id,
            'private_cpu_capacity': self.private_cpu_capacity,
            'public_cpu_capacity': self.public_cpu_capacity
        }

    def process_private_task(self):
        """Process task from private queue (FIFO)"""
        if self.private_queue:
            task = self.private_queue.popleft()  # Get the first task in the private queue
            print(f"Processing task from private queue of MEC {self.id}")
            # Task is processed here with the private CPU capacity
            return True
        return False

    def process_public_tasks(self):
        """Process tasks from public queues (FIFO for each queue)"""
        for queue_id, queue in self.public_queues.items():
            if queue:
                task = queue.popleft()  # Get the first task in the public queue
                print(f"Processing task from public queue {queue_id} of MEC {self.id}")
                # Task is processed here with the public CPU capacity for that queue
                return True
        return False

class Cloud:
    def __init__(self, id, computational_capacity, num_servers):
        self.id = id
        self.computational_capacity = computational_capacity
        self.public_queues = {i: deque() for i in range(num_servers)}  # Public queues for tasks from all MECs (FIFO)

    def get_info(self):
        return {
            'id': self.id,
            'computational_capacity': self.computational_capacity
        }

    def process_public_tasks(self):
        """Process tasks from public queues (FIFO for each queue)"""
        for queue_id, queue in self.public_queues.items():
            if queue:
                task = queue.popleft()  # Get the first task in the public queue
                print(f"Processing task from public queue {queue_id} in Cloud {self.id}")
                # Task is processed here with the Cloud CPU capacity
                return True
        return False

class Environment:
    def __init__(self, num_servers, num_clouds, cloud_cpu_capacity, server_cpu_capacities, folder_path):
        self.num_servers = num_servers
        self.num_clouds = num_clouds  # Number of Clouds
        self.folder_path = folder_path  # Folder where CSV files will be saved
        
        # Create Cloud(s)
        self.clouds = [Cloud(i, cloud_cpu_capacity, num_servers) for i in range(num_clouds)]
        
        # Create MEC servers with the same private and public CPU capacity for all
        self.servers = [MEC(i, server_cpu_capacities, server_cpu_capacities, num_servers) 
                        for i in range(num_servers)]

        # Create folder if it doesn't exist
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def save_environment_csv(self, filename='environment.csv'):
        # Collecting server and cloud info for CSV
        server_data = []
        for server in self.servers:
            server_info = server.get_info()
            server_data.append({
                'Server ID': server_info['id'],
                'Private CPU Capacity': server_info['private_cpu_capacity'],
                'Public CPU Capacity': server_info['public_cpu_capacity']
            })
        
        # Saving server data into CSV in the specified folder
        df_servers = pd.DataFrame(server_data)
        server_file_path = os.path.join(self.folder_path, filename)
        df_servers.to_csv(server_file_path, index=False)
        
        # Saving cloud data into CSV in the specified folder
        cloud_info = [cloud.get_info() for cloud in self.clouds]
        df_cloud = pd.DataFrame(cloud_info)
        cloud_file_path = os.path.join(self.folder_path, 'cloud_info.csv')
        df_cloud.to_csv(cloud_file_path, index=False)

    def load_environment(self, filename='environment.csv'):
        # Loading server data from CSV
        server_file_path = os.path.join(self.folder_path, filename)
        df_servers = pd.read_csv(server_file_path)
        server_data = df_servers.to_dict(orient='records')
        
        # Re-initializing servers with loaded data
        self.servers = [MEC(server['Server ID'], server['Private CPU Capacity'], 
                            server['Public CPU Capacity'], self.num_servers) 
                        for i, server in enumerate(server_data)]

    def get_environment_info(self):
        # Returning environment info including cloud and server data
        environment_info = {
            'clouds': [cloud.get_info() for cloud in self.clouds],
            'servers': [server.get_info() for server in self.servers]
        }
        return environment_info


# Example Usage
num_mec = 20  # Number of MECs
num_cloud = 1  # Number of Clouds
mec_private_cpu_capacities = 0.5 * 10  # Private CPU capacities for each MEC
mec_public_cpu_capacities = 0.5 * 10  # Public CPU capacities for each MEC
cloud_cpu_capacity = 3.0 * 10  # Cloud CPU capacity
folder_path = './simulation_output'  # Folder where the CSVs will be saved

# Initialize environment
env = Environment(num_servers=num_mec, num_clouds=num_cloud, cloud_cpu_capacity=cloud_cpu_capacity,
                  server_cpu_capacities=mec_private_cpu_capacities, folder_path=folder_path)

# Save environment to CSV in the specified folder
env.save_environment_csv()

# Load environment from CSV (for testing)
env2 = Environment(num_servers=num_mec, num_clouds=num_cloud, cloud_cpu_capacity=cloud_cpu_capacity,
                   server_cpu_capacities=mec_private_cpu_capacities, folder_path=folder_path)
env2.load_environment()

# Get environment info
env_info = env2.get_environment_info()
print(env_info)
