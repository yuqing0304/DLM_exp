import re
import matplotlib.pyplot as plt
import numpy as np

# Function to parse the data from a file
def parse_results(file_path):
    data = {}
    current_agent = None
    with open(file_path, 'r') as file:
        content = file.readlines()

    for line in content:
        # Match the agent_num
        agent_match = re.search(r'agent_num: (\d+)', line)
        if agent_match:
            current_agent = f"agent_{agent_match.group(1)}"
            data[current_agent] = {}
        # Match the phase (0, 30, 60) and corresponding values
        phase_match = re.search(r'(\d+): (\{.*?\})', line)
        if phase_match and current_agent:
            phase = int(phase_match.group(1))
            results = eval(phase_match.group(2))  # Convert string to dictionary
            data[current_agent][phase] = results

    return data

# Read and parse results from both files
file_path1 = '512_512_noise0.txt'     # Base noise level
file_path2 = '512_512_noise0.1.txt'  # Higher noise level
file_path3 = '256_512_noise0.txt'    # Lower spk capacity
file_path4 = '256_512_noise0.1.txt'  # Lower spk capacity and higher noise level
# file_path_impa = '512_512_noise0impa.txt'     # Impa condition

file_path_spk1 = '512_512_noise0spk.txt'  # spk60 for noise 0
file_path_spk2 = '512_512_noise0.1spk.txt'  # spk60 for noise 0.1
file_path_spk3 = '256_512_noise0spk.txt'  # spk60 for lower spk capacity
file_path_spk4 = '256_512_noise0.1spk.txt'  # spk60 for lower spk capacity and noise 0.1
# file_path_impa_spk = '512_512_noise0impaspk.txt'  # Impa spk condition


data1 = parse_results(file_path1)
data2 = parse_results(file_path2)
data3 = parse_results(file_path3)
data4 = parse_results(file_path4)
# data_impa = parse_results(file_path_impa)  # Impa data

# Initialize lists to store the percent_short for each phase (spk60, 0, 30, 60) across all agents
percent_short_per_phase_1 = {'spk60': [], 0: [], 30: [], 60: []}
percent_short_per_phase_2 = {'spk60': [], 0: [], 30: [], 60: []}
percent_short_per_phase_3 = {'spk60': [], 0: [], 30: [], 60: []}
percent_short_per_phase_4 = {'spk60': [], 0: [], 30: [], 60: []}
# percent_short_per_phase_impa = {'spk60': [], 0: [], 30: [], 60: []}  # For Impa condition

# Collect percent_short values for each phase across all agents for base noise level
for agent, phase_data in data1.items():
    for phase in [0, 30, 60]:
        if phase in phase_data:
            percent_short = phase_data[phase]['percent_short']
            percent_short_per_phase_1[phase].append(percent_short)

# Collect percent_short values for each phase across all agents for noise 0.1
for agent, phase_data in data2.items():
    for phase in [0, 30, 60]:
        if phase in phase_data:
            percent_short = phase_data[phase]['percent_short']
            percent_short_per_phase_2[phase].append(percent_short)

# Collect percent_short values for each phase across all agents for lower spk capacity
for agent, phase_data in data3.items():
    for phase in [0, 30, 60]:
        if phase in phase_data:
            percent_short = phase_data[phase]['percent_short']
            percent_short_per_phase_3[phase].append(percent_short)

# Collect percent_short values for each phase across all agents for lower spk capacity and noise 0.1
for agent, phase_data in data4.items():
    for phase in [0, 30, 60]:
        if phase in phase_data:
            percent_short = phase_data[phase]['percent_short']
            percent_short_per_phase_4[phase].append(percent_short)

# # Collect percent_short values for Impa condition
# for agent, phase_data in data_impa.items():
#     for phase in [0, 30, 60]:
#         if phase in phase_data:
#             percent_short = phase_data[phase]['percent_short']
#             percent_short_per_phase_impa[phase].append(percent_short)

# Collect percent_short values for spk60 from the new files
def collect_spk_data(file_path, percent_short_per_phase):
    data_spk = parse_results(file_path)
    for agent, phase_data in data_spk.items():
        if 60 in phase_data:
            percent_short = phase_data[60]['percent_short']
            percent_short_per_phase['spk60'].append(percent_short)

collect_spk_data(file_path_spk1, percent_short_per_phase_1)
collect_spk_data(file_path_spk2, percent_short_per_phase_2)
collect_spk_data(file_path_spk3, percent_short_per_phase_3)
collect_spk_data(file_path_spk4, percent_short_per_phase_4)
# collect_spk_data(file_path_impa_spk, percent_short_per_phase_impa)  # For Impa spk

# Calculate the average percent_short for each phase
avg_percent_short_per_phase_1 = [np.mean(percent_short_per_phase_1[phase]) for phase in ['spk60', 0, 30, 60]]
avg_percent_short_per_phase_2 = [np.mean(percent_short_per_phase_2[phase]) for phase in ['spk60', 0, 30, 60]]
avg_percent_short_per_phase_3 = [np.mean(percent_short_per_phase_3[phase]) for phase in ['spk60', 0, 30, 60]]
avg_percent_short_per_phase_4 = [np.mean(percent_short_per_phase_4[phase]) for phase in ['spk60', 0, 30, 60]]
# avg_percent_short_per_phase_impa = [np.mean(percent_short_per_phase_impa[phase]) for phase in ['spk60', 0, 30, 60]]  # For Impa

# Calculate the standard deviation and standard error of the mean (SEM) for each phase
std_per_phase_1 = [np.std(percent_short_per_phase_1[phase]) for phase in ['spk60', 0, 30, 60]]
std_per_phase_2 = [np.std(percent_short_per_phase_2[phase]) for phase in ['spk60', 0, 30, 60]]
std_per_phase_3 = [np.std(percent_short_per_phase_3[phase]) for phase in ['spk60', 0, 30, 60]]
std_per_phase_4 = [np.std(percent_short_per_phase_4[phase]) for phase in ['spk60', 0, 30, 60]]
# std_per_phase_impa = [np.std(percent_short_per_phase_impa[phase]) for phase in ['spk60', 0, 30, 60]]  # For Impa

sem_per_phase_1 = [std / np.sqrt(len(percent_short_per_phase_1[phase])) if len(percent_short_per_phase_1[phase]) > 0 else 0 for std, phase in zip(std_per_phase_1, ['spk60', 0, 30, 60])]
sem_per_phase_2 = [std / np.sqrt(len(percent_short_per_phase_2[phase])) if len(percent_short_per_phase_2[phase]) > 0 else 0 for std, phase in zip(std_per_phase_2, ['spk60', 0, 30, 60])]
sem_per_phase_3 = [std / np.sqrt(len(percent_short_per_phase_3[phase])) if len(percent_short_per_phase_3[phase]) > 0 else 0 for std, phase in zip(std_per_phase_3, ['spk60', 0, 30, 60])]
sem_per_phase_4 = [std / np.sqrt(len(percent_short_per_phase_4[phase])) if len(percent_short_per_phase_4[phase]) > 0 else 0 for std, phase in zip(std_per_phase_4, ['spk60', 0, 30, 60])]
# sem_per_phase_impa = [std / np.sqrt(len(percent_short_per_phase_impa[phase])) if len(percent_short_per_phase_impa[phase]) > 0 else 0 for std, phase in zip(std_per_phase_impa, ['spk60', 0, 30, 60])]  # For Impa

# Calculate the 95% confidence interval (CI) for each phase
ci_95_per_phase_1 = [1.96 * sem for sem in sem_per_phase_1]
ci_95_per_phase_2 = [1.96 * sem for sem in sem_per_phase_2]
ci_95_per_phase_3 = [1.96 * sem for sem in sem_per_phase_3]
ci_95_per_phase_4 = [1.96 * sem for sem in sem_per_phase_4]
# ci_95_per_phase_impa = [1.96 * sem for sem in sem_per_phase_impa]  # For Impa


# Create the plot
x = np.arange(4)  # Four phases: spk60, 0, 30, 60
width = 0.15  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))



# Bars for noise 0 (base level) - very light blue
bars1 = ax.bar(x - 2*width, avg_percent_short_per_phase_1, width, yerr=ci_95_per_phase_1, capsize=5, color='#dbe9f6', label='Baseline') # 'Baseline communication'

# Bars for noise 0.1 - light blue
bars2 = ax.bar(x - width, avg_percent_short_per_phase_2, width, yerr=ci_95_per_phase_2, capsize=5, color='#b3d3ea', label='Noise 0.1')

# Bars for lower spk capacity - slightly darker light blue
bars3 = ax.bar(x, avg_percent_short_per_phase_3, width, yerr=ci_95_per_phase_3, capsize=5, color='#8cbede', label='Lower spk capacity')

# Bars for lower spk capacity and noise 0.1 - medium-light blue
bars4 = ax.bar(x + width, avg_percent_short_per_phase_4, width, yerr=ci_95_per_phase_4, capsize=5, color='#6499b7', label='Lower spk capacity & Noise 0.1')

# # Bars for Impa condition - slightly darker medium-light blue
# bars_impa = ax.bar(x + 2*width, avg_percent_short_per_phase_impa, width, yerr=ci_95_per_phase_impa, capsize=5, color='#457b9d', label='Impa lst')

# Add a horizontal reference line at 50%
ax.axhline(y=50, color='lightgrey', linestyle='--', linewidth=2)

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Epochs', fontsize=14)
ax.set_ylabel('% Local Dependency Utterances', fontsize=14)
ax.set_title('Verb Final, Skewed Mixed', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(['Spk60', 'Comm1', 'Comm30', 'Comm60'], fontsize=12)
ax.legend(fontsize=12, frameon=False)
ax.set_ylim(0, 100)  # Set y-axis limits

# Increase font size for ticks
ax.tick_params(axis='both', which='major', labelsize=12)

# Remove upper and right spines for a cleaner look
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Display the percentage on top of the bars, placed significantly higher
def add_value_labels(bars):
    """Add value labels on top of the bars."""
    for bar in bars:
        height = bar.get_height()
        # ax.annotate(f'{height:.2f}%', 
        #             xy=(bar.get_x() + bar.get_width() / 2, height), 
        #             xytext=(0, 12),  # 12 points vertical offset
        #             textcoords="offset points", 
        #             ha='center', va='bottom', fontsize=10)

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)
add_value_labels(bars4)
# add_value_labels(bars_impa)

# Adjust layout for better fitting
plt.tight_layout()
plt.savefig('comm_final_rnn.pdf', dpi=300)  # Save with higher resolution
plt.show()

