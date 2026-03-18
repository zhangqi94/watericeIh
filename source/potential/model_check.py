from mace.calculators import MACECalculator

model_path = "/home/zq/zqcodeml/watericeIh-mc-master/source/potential/macemodel251125/mace_iceIh_128x0e128x1o_r4.5_float32_seed144_cueq.model"

calc = MACECalculator(model_path)

print("num_models =", calc.num_models)
model = calc.models[0]

print("\n=== model parameters ===")
# for name, p in model.named_parameters():
#     print(f"{name:60s}  requires_grad={p.requires_grad}")
    

for name, p in model.named_parameters():
    print(f"{name:60s}  shape={tuple(p.shape)}  requires_grad={p.requires_grad}")
    


