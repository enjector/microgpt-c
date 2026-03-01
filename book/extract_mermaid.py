import re, glob

for f in sorted(glob.glob("*.md")):
    with open(f, "r") as file:
        content = file.read()
    mem = re.findall(r'```mermaid.*?```', content, re.DOTALL)
    if mem:
        print(f"=== {f} ===")
        for m in mem:
            print(m)
