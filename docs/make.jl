using VoronoiSpheres
using Documenter

DocMeta.setdocmeta!(VoronoiSpheres, :DocTestSetup, :(using VoronoiSpheres); recursive=true)

makedocs(;
    modules=[VoronoiSpheres],
    authors="The ClimFlows contributors",
    sitename="VoronoiSpheres.jl",
    format=Documenter.HTML(;
        canonical="https://ClimFlows.github.io/VoronoiSpheres.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ClimFlows/VoronoiSpheres.jl",
    devbranch="main",
)
