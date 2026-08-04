def parse_dump(path):
    result = {}
    with open(path) as f:
        for line in f:
            name, values_str = line.split(":", 1)
            values = [float(v.strip()) for v in values_str.strip().split(",")]
            result[name.strip()] = values
    return result

py = parse_dump("../weight_files/python_output.txt")
c  = parse_dump("../weight_files/c_output.txt")

mismatches = []
for name in py:
    if name not in c:
        mismatches.append(f"{name}: missing in C output")
        continue
    for i, (pv, cv) in enumerate(zip(py[name], c[name])):
        if abs(pv - cv) > 1e-5:
            mismatches.append(f"{name}[{i}]: py={pv}, c={cv}")

if not mismatches:
    print(f"[OK] all {len(py)} tensors match")
else:
    for m in mismatches:
        print(m)
