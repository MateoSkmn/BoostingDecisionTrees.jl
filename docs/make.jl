using BoostingDecisionTrees
using Documenter

DocMeta.setdocmeta!(BoostingDecisionTrees, :DocTestSetup, :(using BoostingDecisionTrees); recursive=true)

makedocs(;
    modules=[BoostingDecisionTrees],
    authors="Mateo Sakoman <mateo.sakoman@campus.tu-berlin.de>",
    sitename="BoostingDecisionTrees.jl",
    format=Documenter.HTML(;
        canonical="https://MateoSkmn.github.io/BoostingDecisionTrees.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MateoSkmn/BoostingDecisionTrees.jl",
    devbranch="master",
)
