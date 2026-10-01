# Install the R packages of script 07 at the reference versions (HonestDiD 0.2.8, jsonlite 2.0.0; see R_packages.txt).
#   Rscript environment/install_R_packages.R
# While these are the current CRAN versions, the CRAN binaries are installed (no compiler needed on macOS and Windows;
# Linux installs from source). Once CRAN has newer versions, the exact versions are installed from the CRAN archive
# with remotes::install_version (source: needs the toolchain of README section 3). Stops unless the installed versions
# are exactly the reference versions. Dependencies are installed at their current CRAN versions.
repos <- "https://cloud.r-project.org"
want <- c(jsonlite = "2.0.0", HonestDiD = "0.2.8")
type <- if (.Platform$pkgType == "source") "source" else "binary"
current <- available.packages(repos = repos, type = type)[, "Version"]
for (p in names(want)) {
  if (!is.na(current[p]) && current[p] == want[[p]]) {
    install.packages(p, repos = repos, type = type)
  } else {
    if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes", repos = repos)
    remotes::install_version(p, version = want[[p]], repos = repos, upgrade = "never")
  }
}
got <- vapply(names(want), function(p) as.character(packageVersion(p)), "")
cat("installed:", paste(names(got), got), "\n")
if (!identical(unname(got), unname(want))) stop("the installed versions differ from the reference versions")
