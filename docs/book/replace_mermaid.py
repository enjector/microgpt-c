import re
import glob

replacements = {
    '7.md': '![Scalar vs SIMD Execution](simd_bw.png)',
    '8.md': '![MHA vs GQA](gqa_bw.png)',
    '9.md': '![Edge AI Research Loop](edge_loop_bw.png)',
    '10.md': '![Flat-String Protocol](flat_string_bw.png)',
    '13.md': '![Project Speculative Roadmap](roadmap_gantt_bw.png)'
}

for f in sorted(glob.glob("*.md")):
    if f in ['README.md', 'DEPENDS.md', 'LATEX.md', "MicroGPT-C_Composable_Intelligence_at_the_Edge.md"]:
        continue
        
    with open(f, "r") as file:
        content = file.read()
    
    # 1. Replace the older Biology Analogy JPG link with the new PNG in 4.md
    if f == '4.md':
        content = content.replace(
            '![The Biological Blueprint for Tiny AI — stem cell differentiation, the Planner-Worker-Judge triad, and the coordination funnel](../organelles/OPA_Biology_Analogy.jpg)',
            '![The Biological Blueprint for Tiny AI — stem cell differentiation, the Planner-Worker-Judge triad, and the coordination funnel](opa_biology_bw.png)'
        )

    # 2. For specific chapters, inject the new image link before we strip the mermaid block
    if f in replacements:
        target_img = replacements[f]
        # Find the first mermaid block and replace it with the target image
        content = re.sub(r'```mermaid.*?```', target_img, content, count=1, flags=re.DOTALL)
    
    # 3. For any remaining mermaid blocks in ANY file (e.g. ones that already have a B&W pic above them), just strip them
    content = re.sub(r'\s*```mermaid.*?```\s*', '\n\n', content, flags=re.DOTALL)

    with open(f, "w") as file:
        file.write(content)

print("Replacement complete.")
