import os
import re
import subprocess
import hashlib

def process_markdown(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    pattern = re.compile(r'```mermaid\n(.*?)\n```', re.DOTALL)
    
    def replacer(match):
        mermaid_code = match.group(1).strip()
        # Create a unique hash for the diagram to cache it
        code_hash = hashlib.md5(mermaid_code.encode('utf-8')).hexdigest()[:8]
        mmd_file = f"mermaid_{code_hash}.mmd"
        png_file = f"mermaid_{code_hash}.png"
        
        if not os.path.exists(png_file):
            print(f"Rendering Mermaid diagram into {png_file}...")
            with open(mmd_file, 'w') as f:
                f.write(mermaid_code)
            
            try:
                subprocess.run([
                    "npx", "-y", "@mermaid-js/mermaid-cli", 
                    "-i", mmd_file, 
                    "-o", png_file,
                    "-s", "2",
                    "-b", "white",
                    "-t", "default"
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"Failed to render {mmd_file}")
                return match.group(0) # On failure, return original block
                
            if os.path.exists(mmd_file):
                os.remove(mmd_file)
            
        return f"\n\n![Mermaid Diagram]({png_file})\n\n"

    new_content = pattern.sub(replacer, content)
    
    with open(file_path, 'w') as f:
        f.write(new_content)

if __name__ == "__main__":
    target = "MicroGPT-C_Composable_Intelligence_at_the_Edge.md"
    if os.path.exists(target):
        process_markdown(target)
