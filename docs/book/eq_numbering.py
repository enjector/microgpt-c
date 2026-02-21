import re
import glob

for f in sorted(glob.glob("*.md")):
    if f in ['README.md', 'DEPENDS.md', 'LATEX.md', "MicroGPT-C_Composable_Intelligence_at_the_Edge.md"]:
        continue
        
    with open(f, "r") as file:
        content = file.read()
    
    # Replace ```math blocks
    content = re.sub(r'```math\n(.*?)\n```', r'\\begin{equation}\n\1\n\\end{equation}', content, flags=re.DOTALL)
    
    # Replace $$ blocks
    content = re.sub(r'\$\$(.*?)\$\$', r'\\begin{equation}\1\\end{equation}', content, flags=re.DOTALL)

    with open(f, "w") as file:
        file.write(content)

print("Equation replacement complete.")
