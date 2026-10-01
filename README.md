# R dashboard (R Markdown / flexdashboard) - alternative interface

Same model and results as the Streamlit app, for RStudio. It reads pre-computed files from `outputs/` (exported by the
notebook, section 12b); no Python is needed to run it.

1. Install once: `install.packages(c("flexdashboard","shiny","dplyr","ggplot2","readr","DT","jsonlite","rmarkdown"))`
2. Open `retail_risk_dashboard.Rmd` in RStudio, set the working directory to this folder
   (Session > Set Working Directory > To Source File Location).
3. Click **Run Document**.

Note: `runtime: shiny` means it must be run, not just knitted. Not re-tested in the build environment (R was unavailable);
the file is your original, unchanged.
