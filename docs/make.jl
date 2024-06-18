using Apollo
using Documenter

DocMeta.setdocmeta!(Apollo, :DocTestSetup, :(using Apollo); recursive=true)

makedocs(;
    modules=[Apollo],
    authors="Joshua Billson",
    sitename="Apollo.jl",
    format=Documenter.HTML(;
        canonical="https://JoshuaBillson.github.io/Apollo.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JoshuaBillson/Apollo.jl",
    devbranch="main",
)
