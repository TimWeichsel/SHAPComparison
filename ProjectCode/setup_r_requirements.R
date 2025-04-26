# Installs the required R-packages
options(repos = c(CRAN = "https://cloud.r-project.org"))
install.packages("data.table", repos = "https://cloud.r-project.org")
install.packages("Rcpp", repos = "https://cloud.r-project.org")
install.packages("Matrix", repos = "https://cloud.r-project.org")