

import json
import pyworld2
import os
from pyworld2.utils import plot_world_variables


# Funcion para cambiar el archivo JSON data para el World2
def update_json(data, brn1, nrun1, fc1, cign1, poln):
      
    for entry in data:
        if "BRN1" in entry:
            entry["BRN1"] = brn1 # BRN - Birth Rate Normal [fraction/year] Base run 0.028
        elif "NRUN1" in entry:
            entry["NRUN1"] = nrun1 # NRUN - Natural-Resource Usage Normal Base run 0.25
        elif "POLN" in entry:
            entry["POLN"] = poln # Pollution Normal [pollution units/person/year].
        elif "FC1" in entry:
            entry["FC1"] = fc1 # FC - Food Coefficient [] Base run 0.8
        elif "CIDN1" in entry:
            entry["CIDN1"] = cign1 # CIDN - Capital-Investment Discard Normal [fraction/year] Base run 0.03
    return data


def run_full(P =[0.04, 1, 1, 0.05,1],input_file =  "functions_switch_default.json", 
                    output_file = "updated_data.json" ):
    
   
    
    if output_file is not None:
        input_file=os.path.join(os.path.dirname(__file__),'pyworld2', input_file)
        output_file=os.path.join(os.path.dirname(__file__), output_file)

        with open(input_file, "r") as file:
            json_data = json.load(file)

        updated_data = update_json(json_data, P[0], P[1], P[2], P[3], P[4])

        with open(output_file, "w") as file:
            json.dump(updated_data, file, indent=4)


    w2_std = pyworld2.World2()
    w2_std.set_state_variables()
    w2_std.set_initial_state()
    w2_std.set_table_functions()
    w2_std.set_switch_functions(output_file)
    w2_std.run()

    return w2_std.aveg_ql()

def run_analisis(P = [0.04, 0.25, 1, 1, 0.05],input_file =  "functions_switch_default.json", 
                    output_file = "updated_data.json" ):
    
   
    
    if output_file is not None:
        input_file=os.path.join(os.path.dirname(__file__),'pyworld2', input_file)
        output_file=os.path.join(os.path.dirname(__file__), output_file)

        with open(input_file, "r") as file:
            json_data = json.load(file)

        updated_data = update_json(json_data, P[0], P[1], P[2], P[3], P[4])

        with open(output_file, "w") as file:
            json.dump(updated_data, file, indent=4)


    w2_std = pyworld2.World2()
    w2_std.set_state_variables()
    w2_std.set_initial_state()
    w2_std.set_table_functions()
    w2_std.set_switch_functions(output_file)
    w2_std.run()

    plot_world_variables(w2_std.time,
                        [w2_std.ql],
                        [ "QL"],
                        [[0,15]],
                        figsize=(7, 4), grid=True,
                        title="Calidad de vida con parámetros")
    
    return w2_std.aveg_ql(),w2_std.ql, w2_std.brn, w2_std.nrun, w2_std.fc, w2_std.cign, w2_std.poln

