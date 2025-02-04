import json
import random
import pyworld2
from pyworld2.utils import plot_world_variables, plt

# Function to update JSON data
def update_json(data, brn1, nrun1, drn1, fc1, cign1):
    for entry in data:
        if "BRN1" in entry:
            entry["BRN1"] = brn1
        elif "NRUN1" in entry:
            entry["NRUN1"] = nrun1
        elif "DRN1" in entry:
            entry["DRN1"] = drn1
        elif "FC1" in entry:
            entry["FC1"] = fc1
        elif "CIGN1" in entry:
            entry["CIGN1"] = cign1
    return data

input_file = "./pyworld2/functions_switch_default.json"

# Leer el archivo JSON y guardar los datos
with open(input_file, "r") as file:
    json_data = json.load(file)

# Asignar nuevos valores a las variables
brn1 = random.uniform(0.1, 1)  # BRN - Birth Rate Normal [fraction/year] Base run 0.028
nrun1 = random.uniform(0.1, 1)  # NRUN - Natural-Resource Usage Normal Base run 0.25
drn1 = random.uniform(0.1, 1)  # DRN - Death Rate Normal [fraction/year] Base run 0.028
fc1 = random.uniform(0.1, 1)  # FC - Food Coefficient [] Base run 0.8
cign1 = random.uniform(0.1, 1)  # CIDN - Capital-Investment Discard Normal [fraction/year] Base run 0.03

# Cambiar los valores de los parametros en los datos
updated_data = update_json(json_data, brn1, nrun1, drn1, fc1, cign1)

# Creacion de un nuevo archivo json con los datos cambiados
output_file = "updated_data.json"
with open(output_file, "w") as file:
    json.dump(updated_data, file, indent=4)

# Scenario 1 - Standard run
w2_std = pyworld2.World2()
w2_std.set_state_variables()
w2_std.set_initial_state()
w2_std.set_table_functions()
w2_std.set_switch_functions()
w2_std.run()
print(f' el ql promedio es {w2_std.aveg_ql()}')
plot_world_variables(w2_std.time,
                     [w2_std.p, w2_std.polr, w2_std.ci, w2_std.ql, w2_std.nr],
                      ["P", "POLR", "CI", "QL", "NR"],
                      [[0, 8e9], [0, 40], [0, 20e9], [0, 2], [0, 1000e9]],
                      figsize=(7, 4), grid=True,
                      title="World2 - Scenario 1 [Standard run]")

# Scenario 2 - Random
w2_std = pyworld2.World2()
w2_std.set_state_variables()
w2_std.set_initial_state()
w2_std.set_table_functions()
w2_std.set_switch_functions(output_file)
w2_std.run()
print(f' el ql promedio es {w2_std.aveg_ql()}')
plot_world_variables(w2_std.time,
                     [w2_std.p, w2_std.polr, w2_std.ci, w2_std.ql, w2_std.nr],
                      ["P", "POLR", "CI", "QL", "NR"],
                      [[0, 8e9], [0, 40], [0, 20e9], [0, 2], [0, 1000e9]],
                      figsize=(7, 4), grid=True,
                      title="World2 - Scenario 1 [Standard run]")
plt.show()

