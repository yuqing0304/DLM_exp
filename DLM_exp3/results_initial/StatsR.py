import csv

# Define communication and speaker files
comm_files = [
    ("1024_1024_noise0.txt", "normal"),
    ("1024_1024_noise0impa.txt", "impa"),
    ("1024_1024_noise0.1impa.txt", "noise_impa"),
]

spk_files = {
    "normal": "1024_1024_noise0spk.txt",
    "impa": "1024_1024_noise0spkimpa.txt",
    "noise_impa": "1024_1024_noise0.1spkimpa.txt",
}

# Parse speaker file to get {seed: spk60}
def parse_spk_file(file_path):
    spk_dict = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("agent_num:"):
            seed = int(line.split(":")[1].strip())
            line_60 = lines[i+3].strip()
            line_60_clean = line_60.split(":", 1)[1].strip()
            spk60 = eval(line_60_clean).get("percent_so_sm")
            spk_dict[seed] = spk60
            i += 4
        else:
            i += 1
    return spk_dict

# Load all speaker data
spk_data = {condition: parse_spk_file(path) for condition, path in spk_files.items()}

# Process communication files and combine with spk60
results = []

for file_path, condition in comm_files:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("agent_num:"):
            seed = int(line.split(":")[1].strip())

            comm0_line = lines[i+1].strip().split(":", 1)[1].strip()
            comm30_line = lines[i+2].strip().split(":", 1)[1].strip()
            comm60_line = lines[i+3].strip().split(":", 1)[1].strip()

            comm0 = eval(comm0_line).get("percent_short")
            comm30 = eval(comm30_line).get("percent_short")
            comm60 = eval(comm60_line).get("percent_short")

            spk60 = spk_data[condition].get(seed, None)  # None if missing

            results.append([seed, comm0, comm30, comm60, spk60, condition])
            i += 4
        else:
            i += 1

# Write to CSV
output_path = "production_initial_R.csv"
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["seed_num", "comm0", "comm30", "comm60", "spk60", "condition"])
    writer.writerows(results)

print(f"✅ CSV file successfully written to: {output_path}")
