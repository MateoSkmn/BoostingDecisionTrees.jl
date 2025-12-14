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
        "Getting started" => "getting-started.md",
        "Documentation" => "index.md"
    ],
)

deploydocs(;
    repo="github.com/MateoSkmn/BoostingDecisionTrees.jl",
    devbranch="master",
)
