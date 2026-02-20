#!/bin/bash

# Configuration
OUTPUT_MD="MicroGPT-C_Composable_Intelligence_at_the_Edge.md"
OUTPUT_PDF="MicroGPT-C_Composable_Intelligence_at_the_Edge.pdf"

echo "Initializing Book Builder..."

# 1. Reset the output file
rm -f "$OUTPUT_MD"

# 2. Loop through numeric chapters (0-99) and Appendices (A-Z)
# This automatically finds 1.md, 2.md, A.md etc. and sorts them naturally.
for f in $(ls *.md | sort -V); do
  # Skip the output file itself, README, DEPENDS, and LATEX files if present
  if [[ "$f" == "$OUTPUT_MD" || "$f" == "README.md" || "$f" == "DEPENDS.md" || "$f" == "LATEX.md" ]]; then continue; fi
  
  echo "Merging Chapter: $f"
  cat "$f" >> "$OUTPUT_MD"
  
  # Add the separator you used in your original script
  echo $'\n---\n' >> "$OUTPUT_MD" 
done

# 3. Cleanup Citations
# Removes [cite] tags which are common in Gemini outputs
echo "Cleaning citations..."
sed -i '' 's/\[cite[^]]*\]//g' "$OUTPUT_MD"

# 3.5. Replace Unicode box-drawing characters with ASCII equivalents
# This prevents LaTeX errors with pdfLaTeX (which doesn't handle Unicode well)
echo "Replacing Unicode characters for LaTeX compatibility..."
sed -i '' \
    -e 's/│/|/g' \
    -e 's/├/|/g' \
    -e 's/└/`/g' \
    -e 's/┌/+/g' \
    -e 's/┐/+/g' \
    -e 's/┘/+/g' \
    -e 's/─/-/g' \
    "$OUTPUT_MD"

# 4. Check dependencies
if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc not found. Please install it with: brew install pandoc"
    exit 1
fi

# 5. Generate PDF with proper LaTeX math rendering
# Try LaTeX first (best quality), fall back to HTML+MathJax if LaTeX not available
echo "Generating PDF with LaTeX math support..."

# Prefer Unicode-capable engines (XeLaTeX/LuaLaTeX) over pdfLaTeX for better Unicode support
if command -v xelatex &> /dev/null; then
    PDF_ENGINE="xelatex"
elif command -v lualatex &> /dev/null; then
    PDF_ENGINE="lualatex"
elif command -v pdflatex &> /dev/null; then
    PDF_ENGINE="pdflatex"
else
    # No LaTeX engine - use HTML with MathJax and convert to PDF
    echo "No LaTeX engine found. Using HTML with MathJax..."
    HTML_FILE="${OUTPUT_MD%.md}.html"
    
    # Generate HTML with MathJax
    pandoc "$OUTPUT_MD" \
        --from markdown \
        --to html \
        --mathjax \
        --standalone \
        -V geometry:margin=1in \
        -o "$HTML_FILE"
    
    # Try to convert HTML to PDF
    if command -v wkhtmltopdf &> /dev/null; then
        echo "Converting HTML to PDF using wkhtmltopdf..."
        wkhtmltopdf --page-size A4 "$HTML_FILE" "$OUTPUT_PDF"
        rm -f "$HTML_FILE"
        echo "Success! Created $OUTPUT_PDF"
        exit 0
    elif command -v weasyprint &> /dev/null; then
        echo "Converting HTML to PDF using weasyprint..."
        weasyprint "$HTML_FILE" "$OUTPUT_PDF"
        rm -f "$HTML_FILE"
        echo "Success! Created $OUTPUT_PDF"
        exit 0
    else
        echo "Error: No PDF converter found. Please install one of:"
        echo "  Option 1 (recommended): brew install --cask basictex"
        echo "    Then: sudo tlmgr update --self && sudo tlmgr install collection-fontsrecommended"
        echo "  Option 2: brew install wkhtmltopdf"
        echo "  Option 3: pip install weasyprint"
        echo ""
        echo "HTML file saved as: $HTML_FILE (you can open it in a browser and print to PDF)"
        exit 1
    fi
fi

# Use LaTeX engine for PDF generation
# For XeLaTeX/LuaLaTeX, add Unicode font support
if [[ "$PDF_ENGINE" == "xelatex" || "$PDF_ENGINE" == "lualatex" ]]; then
    pandoc "$OUTPUT_MD" \
        --from markdown \
        --to pdf \
        --pdf-engine="$PDF_ENGINE" \
        -V geometry:margin=1in \
        -V papersize=a4 \
        -V mainfont="Times New Roman" \
        -V monofont="Courier New" \
        -o "$OUTPUT_PDF" \
        2>&1 | grep -v "Package hyperref Warning" || true
else
    # For pdfLaTeX, try to handle Unicode (may still have issues)
    pandoc "$OUTPUT_MD" \
        --from markdown \
        --to pdf \
        --pdf-engine="$PDF_ENGINE" \
        -V geometry:margin=1in \
        -V papersize=a4 \
        -o "$OUTPUT_PDF" \
        2>&1 | grep -v "Package hyperref Warning" || true
fi

echo "Success! Created $OUTPUT_PDF"