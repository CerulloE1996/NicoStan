## ---- NicoStan local administrator installer ---------------------------------

## Source this file directly from RStudio to install or reinstall NicoStan.
## The complete build runs automatically in a clean R process, without loading
## or unloading packages in the calling session. Source it again to reinstall.
## To define the functions without installing, first set
## options(NicoStan.admin.autorun = FALSE).


.nicostan_admin_script_path <- function() {

    source_files <- vapply(sys.frames(), function(frame) {
        value <- frame$ofile
        if (is.null(value)) "" else as.character(value)
    }, character(1L))
    source_files <- source_files[nzchar(source_files)]

    if (length(source_files) > 0L) {
        return(normalizePath(tail(source_files, 1L), mustWork = TRUE))
    }

    command_args <- commandArgs(trailingOnly = FALSE)
    file_args <- sub("^--file=", "", command_args[grepl("^--file=", command_args)])
    if (length(file_args) > 0L) {
        return(normalizePath(tail(file_args, 1L), mustWork = TRUE))
    }

    ""
}


.nicostan_admin_find_source <- function(script_path,
                                        source_root = NULL) {

    env_source_root <- Sys.getenv("NICOSTAN_SOURCE_ROOT", unset = "")
    requested_root <- source_root
    if (is.null(requested_root) && nzchar(env_source_root)) {
        requested_root <- env_source_root
    }

    if (!is.null(requested_root) && nzchar(requested_root)) {
        requested_root <- normalizePath(requested_root, mustWork = FALSE)
        if (!file.exists(file.path(requested_root, "DESCRIPTION")) ||
            !file.exists(file.path(requested_root, "inst", "NicoStan", "DESCRIPTION"))) {
            stop(paste0("NICOSTAN_SOURCE_ROOT is not a NicoStan source package: ", requested_root))
        }
        return(requested_root)
    }

    search_roots <- c(
        if (nzchar(script_path)) dirname(script_path) else character(),
        getwd(),
        file.path(path.expand("~"), "Documents", "Work", "PhD_work", "R_packages", "NicoStan")
    )

    for (search_root in unique(search_roots)) {
        candidate <- normalizePath(search_root, mustWork = FALSE)
        for (iteration in seq_len(8L)) {
            if (file.exists(file.path(candidate, "DESCRIPTION")) &&
                file.exists(file.path(candidate, "inst", "NicoStan", "DESCRIPTION"))) {
                return(candidate)
            }
            parent <- dirname(candidate)
            if (identical(parent, candidate)) break
            candidate <- parent
        }
    }

    stop(paste0(
        "Could not locate the NicoStan source package. Source this file from ",
        "R_packages/NicoStan/inst/examples or set NICOSTAN_SOURCE_ROOT."
    ))
}


.nicostan_admin_r_executable <- function() {

    if (.Platform$OS.type == "windows") {
        file.path(R.home("bin"), "R.exe")
    } else {
        file.path(R.home("bin"), "R")
    }
}


.nicostan_admin_install_outer <- function(source_root,
                                          lib) {

    status <- system2(
        command = .nicostan_admin_r_executable(),
        args = c(
            "CMD",
            "INSTALL",
            paste0("--library=", shQuote(lib)),
            "--no-test-load",
            "--preclean",
            shQuote(source_root)
        )
    )

    if (!identical(as.integer(status), 0L)) {
        stop(paste0("The outer NicoStan source installation failed with status ", status, "."))
    }

    invisible(TRUE)
}


.nicostan_admin_run_clean <- function(arguments) {

    work_dir <- tempfile("NicoStan_admin_install_")
    dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
    on.exit(unlink(work_dir, recursive = TRUE, force = TRUE), add = TRUE)

    config_file <- file.path(work_dir, "install_config.rds")
    child_script <- file.path(work_dir, "install.R")
    saveRDS(list(arguments = arguments,
                 library_paths = .libPaths(),
                 script = file.path(arguments$source_root, "inst", "examples", "NicoStan_admin_install.R")),
            config_file)
    writeLines(c(
        "config <- readRDS(commandArgs(trailingOnly = TRUE)[[1L]])",
        "options(NicoStan.admin.autorun = FALSE)",
        ".libPaths(unique(c(config$arguments$lib, config$library_paths, .libPaths())))",
        "Sys.setenv(R_LIBS = paste(.libPaths(), collapse = .Platform$path.sep))",
        "source(config$script, local = TRUE)",
        "do.call(.nicostan_admin_install_worker, config$arguments)"
    ), child_script)

    rscript <- file.path(R.home("bin"), if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
    status <- system2(command = rscript,
                      args = c("--vanilla", shQuote(child_script), shQuote(config_file)))
    if (!identical(as.integer(status), 0L)) {
        stop(paste0("NicoStan installation failed. See the build error above (status ", status, ")."), call. = FALSE)
    }

    invisible(TRUE)
}


#' Install NicoStan from the local outer and inner source packages
#'
#' @param source_root Path to `R_packages/NicoStan`.  The default searches from
#'   this script and then the standard PhD-work package directory.
#' @param lib Target R package library.  The `NICOSTAN_INSTALL_LIB` environment
#'   variable overrides the default first library.
#' @param CUSTOM_FLAGS Optional named list passed to `NicoStan::install_NicoStan`.
#' @return Invisibly, the target library path.
run_NicoStan_admin_install <- function(source_root = NULL,
                                       lib = NULL,
                                       CUSTOM_FLAGS = NULL) {

    script_path <- .nicostan_admin_script_path()
    source_root <- .nicostan_admin_find_source(script_path = script_path,
                                               source_root = source_root)

    env_library <- Sys.getenv("NICOSTAN_INSTALL_LIB", unset = "")
    if (is.null(lib) && nzchar(env_library)) {
        lib <- env_library
    }
    if (is.null(lib)) {
        lib <- .libPaths()[1L]
    }
    if (!is.character(lib) || length(lib) != 1L || !nzchar(lib)) {
        stop("lib must be one non-empty directory path.")
    }

    lib <- normalizePath(lib, mustWork = FALSE)
    dir.create(lib, recursive = TRUE, showWarnings = FALSE)

    package_was_loaded <- any(c("BayesMVP", "NicoStan") %in% loadedNamespaces())
    message("Installing NicoStan in a clean R process. Build output follows below.")
    .nicostan_admin_run_clean(arguments = list(
        source_root = source_root,
        lib = lib,
        CUSTOM_FLAGS = CUSTOM_FLAGS
    ))

    message(paste0("NicoStan installation completed in: ", lib))
    if (package_was_loaded) {
        message("Restart R when you are ready to use the new build; the current session still has its previous package loaded.")
    }

    invisible(lib)
}


.nicostan_admin_install_worker <- function(source_root,
                                          lib,
                                          CUSTOM_FLAGS) {

    ## Keep the installer shell out of the target library. R CMD INSTALL can
    ## then preserve the previous compiled package if the new build fails.
    outer_library <- tempfile("NicoStan_installer_shell_")
    dir.create(outer_library, recursive = TRUE, showWarnings = FALSE)
    on.exit(unlink(outer_library, recursive = TRUE, force = TRUE), add = TRUE)
    .nicostan_admin_install_outer(source_root = source_root, lib = outer_library)

    library("NicoStan", lib.loc = outer_library)
    NicoStan::install_NicoStan(
        CUSTOM_FLAGS = CUSTOM_FLAGS,
        lib = lib,
        force = TRUE
    )

    invisible(lib)
}


if (isTRUE(getOption("NicoStan.admin.autorun", TRUE))) {
    run_NicoStan_admin_install()
}
