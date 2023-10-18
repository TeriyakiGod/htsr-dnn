## Prerequisites

To work with LaTeX documents in this repository, you'll need:

- **Visual Studio Code (VS Code)**
- **Linux/Docker**

## First option of Installation [Linux]

### 1. LaTeX (TeX Live):

- Install TeX Live: `sudo apt install texlive`

### 2. LaTeXmk (Optional, but recommended):

- Install LaTeXmk: `sudo apt install latexmk`

### 3. LaTeX Extension for VS Code:

- Install the "LaTeX Workshop" extension in VS Code.

## Second option of Installation [Docker]

1. **Pull the LaTeX Docker Image**: LaTeX can be run inside a Docker container by using a pre-built Docker image. You can pull a LaTeX Docker image from the official Docker Hub repository using the following command:

   ```bash
   docker pull blang/latex:ctanfull
   ```
2. Create a Directory for Your LaTeX Documents
  ```bash
   mkdir ~/my_latex_documents
   ```
3. Mount your directory
   ```bash
   docker run --rm -it -v ~/my_latex_documents:/data blang/latex:ctanfull pdflatex your_document.tex
   ```
   This command will run pdflatex to compile your document. The compiled PDF file will be in your my_latex_documents directory.

MiKTeX is a modern TeX distribution for Windows, Linux and macOS.




## Documentation
[Latex documentation](https://www.latex-project.org/help/documentation/clsguide.pdf)
