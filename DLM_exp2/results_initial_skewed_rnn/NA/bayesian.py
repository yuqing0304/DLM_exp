import re
import pandas as pd

# Function to parse results from a file and extract agent_num (seed)
def parse_results(file_path):
    data = []
    with open(file_path, 'r') as file:
        content = file.readlines()

    current_agent = None
    for line in content:
        agent_match = re.search(r'agent_num: (\d+)', line)
        if agent_match:
            current_agent = int(agent_match.group(1))  # Store agent_num (seed)
        phase_match = re.search(r'(\d+): (\{.*?\})', line)
        if phase_match and current_agent is not None:
            phase = int(phase_match.group(1))
            results = eval(phase_match.group(2))  # Convert string to dictionary
            data.append({
                "seed": current_agent,
                "phase": phase,
                **results  # Unpack dictionary values
            })

    return pd.DataFrame(data)  # Return as a DataFrame for easy handling

# Function to extract phase "60" and label it as "spk60"
def parse_spk_results(file_path):
    data = []
    with open(file_path, 'r') as file:
        content = file.readlines()

    current_agent = None
    for line in content:
        agent_match = re.search(r'agent_num: (\d+)', line)
        if agent_match:
            current_agent = int(agent_match.group(1))  # Store agent_num (seed)
        phase_match = re.search(r'(\d+): (\{.*?\})', line)
        if phase_match and current_agent is not None:
            phase = int(phase_match.group(1))
            if phase == 60:  # Only extract phase 60 and label as "spk60"
                results = eval(phase_match.group(2))
                data.append({
                    "seed": current_agent,
                    "phase": "spk60",  # Label it explicitly
                    **results
                })

    return pd.DataFrame(data)

# Read and parse standard results
df1 = parse_results('512_512_noise0.txt')
df2 = parse_results('512_512_noise0.1.txt')
df3 = parse_results('256_512_noise0.txt')
df4 = parse_results('256_512_noise0.1.txt')

# Assign condition labels
df1["condition"] = "BaseNoise"
df2["condition"] = "HighNoise"
df3["condition"] = "LowSpkCapacity"
df4["condition"] = "LowSpkCapacity_HighNoise"

# Read and parse speaker results (only extracting phase "spk60")
df_spk1 = parse_spk_results('512_512_noise0spk.txt')
df_spk2 = parse_spk_results('512_512_noise0.1spk.txt')
df_spk3 = parse_spk_results('256_512_noise0spk.txt')
df_spk4 = parse_spk_results('256_512_noise0.1spk.txt')

# Assign condition labels for speaker files
df_spk1["condition"] = "BaseNoise"
df_spk2["condition"] = "HighNoise"
df_spk3["condition"] = "LowSpkCapacity"
df_spk4["condition"] = "LowSpkCapacity_HighNoise"

# Combine all datasets
df_all = pd.concat([df1, df2, df3, df4, df_spk1, df_spk2, df_spk3, df_spk4])

# Save to CSV for Bayesian modeling
df_all.to_csv("bayesian_input_data.csv", index=False)

print("Data saved successfully to bayesian_input_data.csv")


# # Function to parse the data and store agent_num (seed)
# def parse_results(file_path):
#     data = []
#     with open(file_path, 'r') as file:
#         content = file.readlines()

#     current_agent = None
#     for line in content:
#         agent_match = re.search(r'agent_num: (\d+)', line)
#         if agent_match:
#             current_agent = int(agent_match.group(1))  # Store agent_num (seed)
#         phase_match = re.search(r'(\d+): (\{.*?\})', line)
#         if phase_match and current_agent is not None:
#             phase = int(phase_match.group(1))
#             results = eval(phase_match.group(2))  # Convert string to dictionary
#             data.append({
#                 "seed": current_agent,
#                 "phase": phase,
#                 **results  # Unpack dictionary values
#             })

#     return pd.DataFrame(data)  # Return as a DataFrame for easy handling

# # Read and parse results from files
# df1 = parse_results('512_512_noise0.txt')
# df2 = parse_results('512_512_noise0.1.txt')
# df3 = parse_results('256_512_noise0.txt')
# df4 = parse_results('256_512_noise0.1.txt')

# # Add condition labels for Bayesian modeling
# df1["condition"] = "BaseNoise"
# df2["condition"] = "HighNoise"
# df3["condition"] = "LowSpkCapacity"
# df4["condition"] = "LowSpkCapacity_HighNoise"

# # Combine all datasets
# df_all = pd.concat([df1, df2, df3, df4])

# # Save to CSV for Bayesian modeling
# df_all.to_csv("bayesian_input_data.csv", index=False)
