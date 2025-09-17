# define the path to the R script
r_libs_dir = Sys.getenv("R_LIBS_USER") #get the path to R library
if (dir.exists(r_libs_dir) == FALSE) { #check if R path exists
  dir.create(r_libs_dir, recursive = TRUE) #create path to R library
}
.libPaths(Sys.getenv("R_LIBS_USER"))  #define the library path

# Installs the required R-packages
options(repos = c(CRAN = "https://cloud.r-project.org")) #set CRAN mirror
install.packages("data.table", repos = "https://cloud.r-project.org")
install.packages("Rcpp", repos = "https://cloud.r-project.org")
install.packages("Matrix", repos = "https://cloud.r-project.org")
install.packages("shapr") # install R package of shapr git