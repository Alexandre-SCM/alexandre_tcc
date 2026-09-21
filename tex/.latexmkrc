# Guarda PDF e arquivos auxiliares dentro de docs/manuscript_latex/build.
$out_dir = 'build';
$aux_dir = 'build';

# Usa pdflatex como motor principal de compilacao.
$pdf_mode = 1;

# Permite ao BibTeX encontrar a bibliografia no diretorio atual
# e em docs/references.
$ENV{'BIBINPUTS'} = '.:../references:';
