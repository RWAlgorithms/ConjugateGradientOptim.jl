using Documenter
using MinimalistConjGradOptim

makedocs(
    sitename = "MinimalistConjGradOptim",
    format = Documenter.HTML(),
    modules = [MinimalistConjGradOptim]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
