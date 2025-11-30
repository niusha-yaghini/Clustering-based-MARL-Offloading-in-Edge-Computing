import pandas as pd
import numpy as np
import os
import json
from collections import deque

class MEC:
    def __init__(self, id, private_cpu_capacity, public_cpu_capacity, num_servers):
        self.id = id
        self.private_cpu_capacity = private_cpu_capacity  # Private CPU processing capacity
        self.public_cpu_capacity = public_cpu_capacity    # Public CPU processing capacity
        self.private_queue = deque()  # Private queue for local tasks (FIFO)
        # Public queues for tasks coming from other MECs (FIFO)
        self.public_queues = {i: deque() for i in range(num_servers) if i != self.id}

    def get_info(self):
        return {
            'id': self.id,
            'private_cpu_capacity': self.private_cpu_capacity,
            'public_cpu_capacity': self.public_cpu_capacity
        }

    def process_private_task(self):
        """Process one task from the private queue (FIFO)."""
        if self.private_queue:
            task = self.private_queue.popleft()  # Get the first task in the private queue
            print(f"Processing task from private queue of MEC {self.id}")
            # Task would be processed here using the private CPU capacity
            return True
        return False

    def process_public_tasks(self):
        """Process one task from the public queues (FIFO in each queue)."""
        for queue_id, queue in self.public_queues.items():
            if queue:
                task = queue.popleft()  # Get the first task in the public queue
                print(f"Processing task from public queue {queue_id} of MEC {self.id}")
                # Task would be processed here using the public CPU capacity
                return True
        return False

class Cloud:
    def __init__(self, id, computational_capacity, num_servers):
        self.id = id
        self.computational_capacity = computational_capacity
        # Public queues for tasks from all MECs (FIFO)
        self.public_queues = {i: deque() for i in range(num_servers)}

    def get_info(self):
        return {
            'id': self.id,
            'computational_capacity': self.computational_capacity
        }

    def process_public_tasks(self):
        """Process one task from the public queues (FIFO in each queue)."""
        for queue_id, queue in self.public_queues.items():
            if queue:
                task = queue.popleft()  # Get the first task in the public queue
                print(f"Processing task from public queue {queue_id} in Cloud {self.id}")
                # Task would be processed here using the cloud computational capacity
                return True
        return False
    
class Environment:
    def __init__(self, config):
        """
        Initialize the environment from a configuration dictionary.

        Expected config structure:
        {
            "num_mecs": <int>,
            "num_clouds": <int>,
            "folder_path": "<path string>",
            "mecs": [
                {
                    "id": <int>,
                    "private_cpu_capacity": <float>,
                    "public_cpu_capacity": <float>
                },
                ...
            ],
            "clouds": [
                {
                    "id": <int>,
                    "computational_capacity": <float>
                },
                ...
            ]
        }
        """
        self.num_servers = config["num_mecs"]
        self.num_clouds = config["num_clouds"]
        self.folder_path = config["folder_path"]

        # Create cloud instances based on config
        cloud_configs = config.get("clouds", [])
        self.clouds = [
            Cloud(cloud_conf["id"], cloud_conf["computational_capacity"], self.num_servers)
            for cloud_conf in cloud_configs
        ]

        # Create MEC server instances based on config
        mec_configs = config.get("mecs", [])
        # Optionally we could validate that len(mec_configs) == self.num_servers
        self.servers = [
            MEC(mec_conf["id"],
                mec_conf["private_cpu_capacity"],
                mec_conf["public_cpu_capacity"],
                self.num_servers)
            for mec_conf in mec_configs
        ]

        # Create output folder if it does not exist
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def save_environment_csv(self, filename='environment.csv'):
        """Save MEC and Cloud information into CSV files."""
        # Collect server info
        server_data = []
        for server in self.servers:
            server_info = server.get_info()
            server_data.append({
                'Server ID': server_info['id'],
                'Private CPU Capacity': server_info['private_cpu_capacity'],
                'Public CPU Capacity': server_info['public_cpu_capacity']
            })

        # Save server data to CSV
        df_servers = pd.DataFrame(server_data)
        server_file_path = os.path.join(self.folder_path, filename)
        df_servers.to_csv(server_file_path, index=False)

        # Save cloud data to CSV
        cloud_info = [cloud.get_info() for cloud in self.clouds]
        df_cloud = pd.DataFrame(cloud_info)
        cloud_file_path = os.path.join(self.folder_path, 'cloud_info.csv')
        df_cloud.to_csv(cloud_file_path, index=False)

    def load_environment(self, filename='environment.csv'):
        """
        Load server information from CSV and recreate MEC server objects.
        Cloud configuration is not reloaded here (only MECs).
        """
        server_file_path = os.path.join(self.folder_path, filename)
        df_servers = pd.read_csv(server_file_path)
        server_data = df_servers.to_dict(orient='records')

        # Re-initialize MEC servers using the loaded data
        self.servers = [
            MEC(server['Server ID'],
                server['Private CPU Capacity'],
                server['Public CPU Capacity'],
                self.num_servers)
            for server in server_data
        ]

    def get_environment_info(self):
        """Return a dictionary containing information about clouds and servers."""
        environment_info = {
            'clouds': [cloud.get_info() for cloud in self.clouds],
            'servers': [server.get_info() for server in self.servers]
        }
        return environment_info


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    # Path to the JSON configuration file
    config_path = "environment_config.json"

    # Load configuration from JSON file
    with open(config_path, "r") as f:
        config = json.load(f)

    # Initialize environment from JSON config
    env = Environment(config)

    # Save environment information to CSV files
    env.save_environment_csv()

    # Create a new Environment instance using the same config
    env2 = Environment(config)
    # Load server information from CSV into the new environment instance
    env2.load_environment()

    # Retrieve and print environment information
    env_info = env2.get_environment_info()
    print(env_info)